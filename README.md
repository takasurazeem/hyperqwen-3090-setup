# Swift-1.5 Qwen3.8-27B at 262k context on one RTX 3090 — two stacks, one recommendation

A working setup for serving the **ukisai/Swift-1.5-Qwen3.8-27b** fine-tune (AWQ INT4)
at the model's **full native 262,144-token context on one RTX 3090 (24 GB)**, measured
on the same box with **two serving stacks**:

- **[HyperQwen](https://github.com/syv-ai/HyperQwen)** — a patched vLLM, with the
  **MTP** speculative drafter, prefix caching, and the vision tower on. This is what
  the repo originally documented — hence the name.
- **[llamAmpere](https://github.com/JakeATX/llamAmpere)** — a llama.cpp fork for
  Ampere cards (RTX 3090 / 3090 Ti), **v0.4** (MIT), whose central feature is that it
  **uses the Qwen3.8 MTP head**, which stock llama.cpp discards.

**The 2026-10-03 measurements change the recommendation.** On this box, llamAmpere now
beats the patched-vLLM stack on the 5-task set (35.05 s vs 36.82 s for
HyperQwen + Swift 1.5) and, dramatically, on prefix caching at depth (1398.8x warm
speedup at a 252k prompt vs HyperQwen's 256x). Since an agent re-sends its accumulating
history every turn, the prefix cache — not raw decode — dominates turn cost at depth.
**llamAmpere is the better default for long-horizon / agent work.** HyperQwen remains
the stack with the verified 262k needle retrieval (207k prompt), the vision tower, and
the measured 95.1%-hit-rate agent run. The full numbers are in
[BENCHMARKS.md](BENCHMARKS.md); the exact environment values for both stacks are in
[CONFIG.md](CONFIG.md).

## Read this first: the MTP-head correction

**Any earlier llama.cpp figure for this model family — from this repo's earlier
measurements, from llama.cpp in general, or from any "3090 + Qwen3.8-27B" comparison —
was measured with the MTP head DISCARDED.** Stock llama.cpp does not use Qwen3.8's MTP
head. On the same GGUF it logs:

```
W model has unused tensor blk.64.attn_norm.weight -- ignoring
W model has unused tensor blk.64.attn_q.weight    -- ignoring
... 15 tensors, 0.22 GiB total, silently thrown away
```

llamAmpere uses the head instead:

```
I speculative: MTP drafter on by default for qwen35 (nextn=1): draft-mtp, n-max 4
I common_speculative_init_result: MTP draft context uses the draft-only vocabulary shortlist 'auto'
I srv load_model: n_slots = 1, n_ctx_slot = 262144
I slot print_timing: draft acceptance = 0.61 (141 accepted / 232 generated), mean len = 3.43
```

"llama.cpp numbers" and "llama.cpp **with MTP** numbers" are different things. Every
stock-llama.cpp figure in these docs is explicitly labeled **MTP discarded**.

## Hardware

| | |
|---|---|
| GPU | one NVIDIA RTX 3090, 24 GB (sm86) |
| Power cap | **280 W** (`sudo nvidia-smi -pl 280`) |

The upstream HyperQwen project benchmarks its reference 3090 at 250 W; we run 280 W.
Absolute rates will not line up with upstream tables for that reason (upstream's own
docs note that on this card sustained throughput is mostly a function of the power cap,
with 200→250→280 W giving ~57.5→85.6→86.7 tok/s in a sustained decode ladder).

## The two stacks

### Stack 1: HyperQwen (patched vLLM)

Stock vLLM (and the model's plain AWQ export) cannot serve this fine-tune at 262k on
one 24 GB card. Two things stand in the way, and both are solved by the same
patched-server + prepare-pipeline combination:

1. **The fine-tune ships its large matrices in bf16.** The AWQ INT4 body is ~16 GB,
   but `lm_head` and `embed_tokens` (vocab 248,320) plus the MTP module are *not*
   part of the int4 quantization — the exported checkpoint carries them in bf16,
   roughly 2.5 GB per embedding matrix. That is ~5 GB of full-precision weight that
   does not shrink when you quantize.
2. **262k of context is real memory.** The model is hybrid: 48 linear-attention
   (Gated DeltaNet) layers plus 16 full-attention layers with 4 KV heads and
   head_dim 256. The KV cache for the 16 attention layers is 2 × 16 × 4 × 256 × 2
   bytes ≈ **8 KiB per token in bf16**, i.e. ~2.1 GB just for 262k of attention KV,
   on top of the recurrent state of the 48 DeltaNet layers and whatever the drafter
   needs.

On 24 GB: ~16 GB weights + ~5 GB bf16 heads + ~2.1 GB of 262k KV + activations and
CUDA graphs does not fit.

**The crux: bf16 head requantization.** HyperQwen's `prepare/` pipeline rewrites the
checkpoint *before serving*: `lm_head`, `embed_tokens`, and the MTP module are
**requantized from bf16 to int8** (group-128), in place, streaming the shard. That
halves the ~5 GB of full-precision head weight to ~2.5 GB, which is what frees the
last few gigabytes needed for the 262k KV pool on a 24 GB card. The same pipeline
also slices a 40,960-row draft head for the MTP drafter.

Two more pieces of the patched server matter for the `CTX=huge` profile we run:

- **KVarN**, HyperQwen's 4/2-bit KV-cache backend, which the `CTX=huge` launcher
  selects so the huge context fits the pool (upstream doc:
  [`docs/long-context.md`](https://github.com/syv-ai/HyperQwen/blob/main/docs/long-context.md)).
- **the speculative-decode and attention patch series** (38+ patches in
  [`patches/`](https://github.com/syv-ai/HyperQwen/tree/main/patches)), which is
  where the MTP path on this hybrid architecture lives.

Everything else about the box is deliberately boring: one card, one container,
one model.

### Stack 2: llamAmpere (llama.cpp fork)

[llamAmpere](https://github.com/JakeATX/llamAmpere) (v0.4, MIT) is a llama.cpp fork
for Ampere (RTX 3090 / 3090 Ti). Its central feature is that it **uses the Qwen3.8
MTP head** — the 15 tensors / 0.22 GiB that stock llama.cpp silently throws away
(see the correction above). The MTP drafter is on by default for `qwen35`
(nextn=1, `draft-mtp`, n-max 4), the context slot is 262,144, and the drafter scored
**0.61 acceptance (141 accepted / 232 generated, mean len 3.43)** on our box.

We built it here with **CUDA 13.2, sm_86, host compiler gcc-15**
(`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`; GCC 16 is rejected by nvcc).

One known defect: the fork's draft-vocabulary shortlist is **not applying** on this
box (log: `draft_vocab_fallback: draft vocabulary shortlist not applied … the draft
scores the full head`). The fork's own README claims +8.45% ± 0.94% from that
shortlist, so **all llamAmpere numbers in this repo are below the fork's intended
configuration.** It is tracked as an open issue in [BENCHMARKS.md](BENCHMARKS.md).

## Which stack wins for which workload

| Workload | Winner on this box | Why (measured) |
|---|---|---|
| Long-horizon agent work (history re-sent every turn, deep context) | **llamAmpere** | prefix cache at depth: **1398.8x** warm speedup at a 252,055-token prompt vs HyperQwen's **256x** at 252k; the cache, not raw decode, dominates turn cost at depth |
| 5-task comparison set | **llamAmpere** | **35.05 s** total — best of five configs (HyperQwen fast+MTP on the *base* model: 35.84 s; HyperQwen + Swift 1.5: 36.82 s) |
| Verified 262k context (needle retrieval) | **HyperQwen** | needle planted at a **207k-token prompt** came back (measured). llamAmpere serves 252,055-token prompts warm (measured) but no needle test is recorded for it |
| Vision | **HyperQwen** | `VISION=1` is on and image content parts are served (measured). llamAmpere vision support: **unverified** |
| Shallow-context decode | **HyperQwen (as measured)** | **139.6 tok/s** at shallow context. llamAmpere's measured decode is at real depth and warm (26.6 / 18.2 tok/s) — a different condition, **not a like-for-like comparison** |
| Measured agent run on an OpenAI-compatible API | **HyperQwen** | one real multi-file coding ticket: **11.1 minutes**, 39 tool calls, ~81k peak context, **95.1% prefix-cache hit rate**. llamAmpere serves an OpenAI-compatible API (llama.cpp standard), but no equivalent agent-run measurement is recorded for it |

Two caveats before you pick: (1) the task-set rows are the finding's "identical 5
tasks, same box, same GGUF/family" — the HyperQwen rows use the base Qwen3.8-27B and
the Swift 1.5 fine-tune respectively, and the finding does not state which checkpoint
the llamAmpere GGUF is (exact quantization: **unverified** in this document); (2)
llamAmpere's numbers are taken with the vocabulary shortlist disabled (open issue
above), so its real margin may be larger, not smaller.

## What works and what does not

**HyperQwen (patched vLLM):**

| Capability | Result |
|---|---|
| 262k context | **verified**: needle-retrieval test at a **207k-token prompt** — the needle came back |
| Decode, shallow context | **139.6 tok/s** |
| Prefix caching (`PREFIX_CACHE=1`) | **95.1% hit rate** over a real agent run (39 tool calls, ~81k peak context) |
| Tool calling | works (used in the agent run above) |
| Vision (`VISION=1`) | on; image content parts are served |
| Real workload | one real multi-file coding ticket: **11.1 minutes** end to end |

Does not work / do not use:

- **The DFlash2 drafter is unstable at `CTX=huge` on this fine-tune.** It dies on
  prompts of roughly **7k tokens or more** in this profile. We do not run it here.
- **MTP is the drafter of record for this stack** (`SPEC=mtp`, `DRAFT_TOKENS=3`).
  Beyond stability, its prefix-cache behaviour is far better than DFlash2's on this
  model — the 95.1% prefix-cache hit rate above is an MTP number. (Upstream
  HyperQwen's `CTX=huge` headline configuration uses DFlash2 with the *stock* model;
  on this fine-tune it is the wrong choice.)

Note the asymmetry: DFlash2 is the faster drafter for the stock model's reproduction
workloads, and upstream recommends it in several profiles. That recommendation does
not transfer to this fine-tune at `CTX=huge` — which is exactly the kind of
per-checkpoint difference HyperQwen's open issue
[#249](https://github.com/syv-ai/HyperQwen/issues/249) ("bringing your own finetune")
is tracking.

**llamAmpere (llama.cpp fork):**

| Capability | Result |
|---|---|
| MTP head | **used** (stock llama.cpp discards it) — the whole point of the fork |
| 262k context slot | `n_ctx_slot = 262144`; 252,055-token prompts served warm (measured) |
| Prefix caching at depth | **90.3x – 1398.8x** warm speedups from 16.8k to 252k prompts (measured) |
| Decode at real depth (warm) | 26.6 tok/s at ~60k context; 18.2 tok/s at ~120k context (measured) |
| Task set (5 tasks) | **35.05 s** total, best of five configs (measured) |
| Vision | **unverified** — no measurement in this repo |

Does not work / open issues:

- **The draft-vocabulary shortlist is not applying** (`draft_vocab_fallback`): the
  draft scores the full head instead of the shortlist. The fork's README claims
  +8.45% ± 0.94% from the shortlist; it is not active here, so the numbers above are
  **below the fork's intended configuration**. Open issue — see
  [BENCHMARKS.md](BENCHMARKS.md#5-open-issue-the-draft-vocabulary-shortlist-is-not-applying).
- **A 262,144-token context REJECTED a 252k-token *prompt* with HTTP 400** during the
  measurement session — the usable prompt ceiling is below the context size. (The
  finding records the rejection without naming the server; the llamAmpere table above
  shows 252,055-token prompts being served, so it is most plausibly the HyperQwen
  server. Which server exactly: **unverified**.)
- Cold-sweep throughput figures (e.g. 2.7 tok/s at ~40k) are **prefill-dominated and
  must not be quoted as decode speed** — see
  [BENCHMARKS.md](BENCHMARKS.md#3-decode-with-mtp-active-at-real-depth-warm).

## How this relates to what is already on GitHub

As of the writing of this README (2026-10-03), we found no published setup that
matches this exact combination — a *fine-tuned* Qwen3.8-27B on a *single* 24 GB 3090
at ≥128k. The closest things we found:

- **[syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen)** — the patched vLLM
  itself. Serves the *stock* Qwen3.8-27B on one 24 GB card (250 W) at 150k–262k
  context. Same server we use; different model, and its `CTX=huge` profile uses
  DFlash2 rather than MTP.
- **Field report
  [#216](https://github.com/syv-ai/HyperQwen/issues/216)** — the same Swift-1.5
  family fine-tune (`ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound`) on HyperQwen,
  but on **2× RTX 3080 20 GB (TP=2)** at 131k, DFlash2. Two cards, not a 3090.
- **Field report
  [#208](https://github.com/syv-ai/HyperQwen/issues/208)** — a Swift AutoRound
  build at **262k** on HyperQwen, but on **2× RTX 3090 NVLink (TP=2)** with the
  DFlash2 drafter and a KVarN + offload tier. Two cards, not 1.5.
- **[TheRealBluesun/HyperQwen-3090ti](https://github.com/TheRealBluesun/HyperQwen-3090ti)**
  — HyperQwen plus 18 further patches on 3090 Ti / 3090 hardware, but the *stock*
  W4A16 model, DFlash2, 128k-class context, two cards.
- **[Ar4ikov/TurboQwen](https://github.com/Ar4ikov/TurboQwen)** — the HyperQwen
  stack packaged for asymmetric-AWQ exports (base and uncensored, not the Swift
  fine-tune). Single-3090 profiles top out at 100k; its 262k numbers need 2×3090
  TP=2.
- **[JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere)** — now the second
  stack documented in this repo (v0.4, MIT). Previously the closest *single-card 3090
  + 262k + Swift-1.5 + MTP* match in existence; it is a **llama.cpp** engine, not
  vLLM, and its MTP-head usage is what changes the recommendation above.

If you run something similar, a field report into
[HyperQwen](https://github.com/syv-ai/HyperQwen/issues) (their
[`docs/reproductions/`](https://github.com/syv-ai/HyperQwen/tree/main/docs/reproductions)
collects them) is the way to add a datapoint.

## Reproducing it

### Stack 1: HyperQwen (patched vLLM)

> **Status of these steps.** The environment values and the measured numbers are
> from our box and are exact. The *commands* below follow the upstream HyperQwen
> flow (clone → `.env` → prepare → compose). The exact prepare-script path and image
> commit for *our specific* checkpoint are **unverified in this document** —
> cross-check against the current upstream README before you start.

**0. Prerequisites.** One RTX 3090 (24 GB), NVIDIA driver with CUDA 12.x, Linux
(upstream docs cover WSL2 variants separately). Docker (the stock path) or a Python
3.12 venv (bare-metal path in upstream `docs/install.md`). ~20 GB free disk for the
prepared model, plus ~10 GB for the container image. Network access to Hugging Face.

**1. Cap the card.**

```bash
sudo nvidia-smi -pl 280        # our cap; upstream's reference is 250 W
```

**2. Clone the server.**

```bash
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen
cp .env.example .env
```

**3. Prepare the fine-tuned model.** Get the SWIFT-1.5 checkpoint in the AWQ INT4
shape (Hugging Face: `ukisai/Swift-1.5-Qwen3.8-27b` — the UkisAI "Swift1.5 27B"
collection; note it is published on HF, not GitHub), then run HyperQwen's one-time
prepare step, which:

- requantizes `lm_head`, `embed_tokens` and the MTP module **to int8** (the crux,
  above),
- slices the MTP **draft head** (40,960 rows),
- regenerates the **draft vocabulary** for *this* checkpoint.

For the container path this is `docker compose run --rm prepare` (the first
`compose up` also runs it and pulls the model into `./models`). For a third-party
export with an asymmetric-AWQ body in a single shard, upstream's
`prepare/quant_heads_stream.py` is the entry point.

Two fine-tune-specific warnings from upstream issue
[#249](https://github.com/syv-ai/HyperQwen/issues/249): the **draft vocabulary**, the
**chat template**, and the **pinned KV-pool constants** must all be regenerated per
checkpoint — reusing the base model's values does not fail loudly, it just quietly
degrades drafter acceptance. (Upstream also fixed the `prepare/` scripts for
AutoRound-derived configs in #246; if you hit `KeyError: 'ignore'` /
`'config_groups'` in the quant scripts, you are on a too-old commit.)

**4. Set the environment.** The full list is in [CONFIG.md](CONFIG.md); the
load-bearing values:

```bash
CTX=huge
MAX_LEN=262144
SPEC=mtp
DRAFT_TOKENS=3
PREFIX_CACHE=1
VISION=1
```

(`CTX=huge` selects the KVarN 4/2-bit KV backend; `MAX_LEN=262144` is the model's
native context; `SPEC=mtp` is the drafter choice explained above.)

**5. Launch.**

```bash
docker compose --profile single up -d
```

The server binds `0.0.0.0:18020` with no auth until you set a key — add
`VLLM_API_KEY=$(openssl rand -hex 24)` to `.env` before exposing it. First boot is
slow (it prepares the model); watch the container logs for the KV-pool size it landed
on.

**6. Verify.**

- `curl http://localhost:18020/v1/models` — the model is up.
- **Needle probe**: embed a known sentence ~200k tokens into an otherwise generated
  document, send it, and check the model quotes it back. (Ours retrieved at 207k.)
- **Prefix cache**: run a repeated conversation and check the prefix-cache counter in
  `/metrics` — ours reads 95.1% over an agent run.

**7. Known failure modes.**

- DFlash2 at `CTX=huge` on this fine-tune: **do not use** (dies on prompts ≥ ~7k,
  see above).
- If the boot dies with an OOM around the split-KV verify buffer or the pool pin, the
  pool constants in the launcher were measured for a different-sized checkpoint —
  regenerate them (upstream #249) rather than guessing at `KV_MEM`.
- If decode is far below the 139.6 tok/s we measured at shallow context, check the
  power cap first (`nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv`): on
  this card a quiet 200 W box measures its cap, not the stack.

### Stack 2: llamAmpere (llama.cpp fork)

> **Status of these steps.** The build values below (CUDA 13.2, sm_86, gcc-15 host
> compiler) are what this box was built with and are exact. The GGUF quantization
> used in our measurements and the exact upstream llama.cpp base commit are
> **unverified in this document** — cross-check against the current llamAmpere
> README before you start.

**0. Prerequisites.** One RTX 3090 (24 GB), Linux, CUDA 13.2 toolkit, CMake, and
**gcc-15** (`g++-15`) as the host compiler. A GGUF of the model family (same
family as the HyperQwen checkpoint; exact quantization: unverified here).

**1. Build.**

```bash
git clone https://github.com/JakeATX/llamAmpere && cd llamAmpere
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15
cmake --build build -j
```

GCC 16 is rejected by nvcc — use gcc-15.

**2. Serve.** The MTP drafter is on by default for `qwen35` (nextn=1, `draft-mtp`,
n-max 4); the context slot is 262,144. Start the llama.cpp server with your GGUF and
a 262k context, then check the log for the MTP lines quoted above (drafter on,
`n_ctx_slot = 262144`) and for draft acceptance.

**3. Verify.**

- The log must show the MTP drafter **on** (`draft-mtp`) — if you see `unused tensor
  … ignoring` lines instead, you are running stock llama.cpp with the MTP head
  discarded, and every number you produce is a discarded-MTP number.
- **Prefix cache**: re-send a long prompt and compare wall time cold vs warm — ours
  hit 1398.8x at a 252,055-token prompt (see [BENCHMARKS.md](BENCHMARKS.md)).
- **Watch for the open issue**: `W draft_vocab_fallback: draft vocabulary shortlist
  not applied …` means the shortlist is off and your numbers are below the fork's
  intended configuration.

## What this repo is (and is not)

It is documentation of one box's configuration, with the measurements that box
produced. It does not fork or patch HyperQwen or llamAmpere, it does not publish a
model, and it claims no numbers we did not measure. Anything we did not measure is
labeled unverified. The repo name keeps the original "HyperQwen" framing; the docs
now cover both stacks, and the recommendation for long-horizon work is llamAmpere.

**Licenses.** The model is Apache-2.0 (UkisAI fine-tune of Qwen); HyperQwen is
Apache-2.0; llamAmpere is MIT. This repo documents all three without modifying any
of them.
