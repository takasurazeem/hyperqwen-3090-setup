# HyperQwen on a single RTX 3090 — Swift-1.5 Qwen3.8-27B at 262k context

A working setup for serving the **ukisai/Swift-1.5-Qwen3.8-27b** fine-tune (AWQ INT4)
at the model's **full native 262,144-token context on one RTX 3090 (24 GB)**, using
**[HyperQwen](https://github.com/syv-ai/HyperQwen)** — a patched vLLM — with the
**MTP** speculative drafter, prefix caching, and the vision tower on.

This repo documents that setup: why the patched vLLM is the load-bearing piece,
what we measured on the box, what does and does not work, and how to reproduce it.
The exact environment values and the measured benchmark table are in
[CONFIG.md](CONFIG.md).

## Why this setup exists

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

### The crux: bf16 head requantization

HyperQwen is not just a config. Its `prepare/` pipeline rewrites the checkpoint
*before serving*: `lm_head`, `embed_tokens`, and the MTP module are **requantized
from bf16 to int8** (group-128), in place, streaming the shard. That halves the
~5 GB of full-precision head weight to ~2.5 GB, which is what frees the last few
gigabytes needed for the 262k KV pool on a 24 GB card. The same pipeline also
slices a 40,960-row draft head for the MTP drafter.

This is the difference between "a 27B W4A16 model" (which fits *somewhere* in 24 GB)
and "a 27B W4A16 model with *its full 262k context* in 24 GB" — the head bytes are
what 262k is up against.

Two more pieces of the patched server matter for the `CTX=huge` profile we run:

- **KVarN**, HyperQwen's 4/2-bit KV-cache backend, which the `CTX=huge` launcher
  selects so the huge context fits the pool (upstream doc:
  [`docs/long-context.md`](https://github.com/syv-ai/HyperQwen/blob/main/docs/long-context.md)).
- **the speculative-decode and attention patch series** (38+ patches in
  [`patches/`](https://github.com/syv-ai/HyperQwen/tree/main/patches)), which is
  where the MTP path on this hybrid architecture lives.

Everything else about the box is deliberately boring: one card, one container,
one model.

## Hardware

| | |
|---|---|
| GPU | one NVIDIA RTX 3090, 24 GB (sm86) |
| Power cap | **280 W** (`sudo nvidia-smi -pl 280`) |

The upstream project benchmarks its reference 3090 at 250 W; we run 280 W.
Absolute rates will not line up with upstream tables for that reason (upstream's
own docs note that on this card sustained throughput is mostly a function of the
power cap, with 200→250→280 W giving ~57.5→85.6→86.7 tok/s in a sustained decode
ladder).

## What works (measured on this box)

| Capability | Result |
|---|---|
| 262k context | **verified**: needle-retrieval test at a **207k-token prompt** — the needle came back |
| Decode, shallow context | **139.6 tok/s** |
| Prefix caching (`PREFIX_CACHE=1`) | **95.1% hit rate** over a real agent run |
| Tool calling | works (used in the agent runs above) |
| Vision (`VISION=1`) | on; image content parts are served |
| Real workload | one real multi-file coding ticket: **11.1 minutes** end to end |

See [CONFIG.md](CONFIG.md#measured-benchmarks) for the table with conditions.

## What does not work

- **The DFlash2 drafter is unstable at `CTX=huge` on this fine-tune.** It dies on
  prompts of roughly **7k tokens or more** in this profile. We do not run it here.
- **MTP is the drafter of record for this setup** (`SPEC=mtp`, `DRAFT_TOKENS=3`).
  Beyond stability, its prefix-cache behaviour is far better than DFlash2's on
  this model — the 95.1% prefix-cache hit rate above is an MTP number. (Upstream
  HyperQwen's `CTX=huge` headline configuration uses DFlash2 with the *stock*
  model; on this fine-tune it is the wrong choice.)

Note the asymmetry: DFlash2 is the faster drafter for the stock model's
reproduction workloads, and upstream recommends it in several profiles. That
recommendation does not transfer to this fine-tune at `CTX=huge` — which is exactly
the kind of per-checkpoint difference HyperQwen's open issue
[#249](https://github.com/syv-ai/HyperQwen/issues/249) ("bringing your own
finetune") is tracking.

## How this relates to what is already on GitHub

As of the writing of this README (2026-10-03), we found no published setup that
matches this exact combination — a *fine-tuned* Qwen3.8-27B on a *single* 24 GB
3090 via a *patched vLLM* at ≥128k. The closest things we found:

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
- **[JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere)** — the closest
  *single-card 3090 + 262k + Swift-1.5 + MTP* match in existence, but it is a
  **llama.cpp** engine (IQ4_XS GGUF), not vLLM.

If you run something similar, a field report into
[HyperQwen](https://github.com/syv-ai/HyperQwen/issues) (their
[`docs/reproductions/`](https://github.com/syv-ai/HyperQwen/tree/main/docs/reproductions)
collects them) is the way to add a datapoint; this README is the fine-tune-side
half of that picture.

## Reproducing it

> **Status of these steps.** The environment values and the measured numbers are
> from our box and are exact. The *commands* below follow the upstream HyperQwen
> flow (clone → `.env` → prepare → compose). The exact prepare-script path and
> image commit for *our specific* checkpoint are **unverified in this document**
> — cross-check against the current upstream README before you start.

### 0. Prerequisites

- One RTX 3090 (24 GB), NVIDIA driver with CUDA 12.x, Linux (upstream docs cover
  WSL2 variants separately).
- Docker (the stock path) or a Python 3.12 venv (bare-metal path in upstream
  `docs/install.md`).
- ~20 GB free disk for the prepared model, plus ~10 GB for the container image.
- Network access to Hugging Face.

### 1. Cap the card

```bash
sudo nvidia-smi -pl 280        # our cap; upstream's reference is 250 W
```

### 2. Clone the server

```bash
git clone https://github.com/syv-ai/HyperQwen && cd HyperQwen
cp .env.example .env
```

### 3. Prepare the fine-tuned model

Get the SWIFT-1.5 checkpoint in the AWQ INT4 shape (Hugging Face:
`ukisai/Swift-1.5-Qwen3.8-27b` — the UkisAI "Swift1.5 27B" collection; note it is
published on HF, not GitHub), then run HyperQwen's one-time prepare step, which:

- requantizes `lm_head`, `embed_tokens` and the MTP module **to int8**
  (the crux, above),
- slices the MTP **draft head** (40,960 rows),
- regenerates the **draft vocabulary** for *this* checkpoint.

For the container path this is `docker compose run --rm prepare` (the first
`compose up` also runs it and pulls the model into `./models`). For a third-party
export with an asymmetric-AWQ body in a single shard, upstream's
`prepare/quant_heads_stream.py` is the entry point.

Two fine-tune-specific warnings from upstream issue
[#249](https://github.com/syv-ai/HyperQwen/issues/249): the **draft vocabulary**,
the **chat template**, and the **pinned KV-pool constants** must all be
regenerated per checkpoint — reusing the base model's values does not fail loudly,
it just quietly degrades drafter acceptance. (Upstream also fixed the `prepare/`
scripts for AutoRound-derived configs in #246; if you hit `KeyError: 'ignore'` /
`'config_groups'` in the quant scripts, you are on a too-old commit.)

### 4. Set the environment

The full list is in [CONFIG.md](CONFIG.md); the load-bearing values:

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

### 5. Launch

```bash
docker compose --profile single up -d
```

The server binds `0.0.0.0:18020` with no auth until you set a key — add
`VLLM_API_KEY=$(openssl rand -hex 24)` to `.env` before exposing it. First boot
is slow (it prepares the model); watch the container logs for the KV-pool size it
landed on.

### 6. Verify

- `curl http://localhost:18020/v1/models` — the model is up.
- **Needle probe**: embed a known sentence ~200k tokens into an otherwise
  generated document, send it, and check the model quotes it back. (Ours
  retrieved at 207k.)
- **Prefix cache**: run a repeated conversation and check the prefix-cache
  counter in `/metrics` — ours reads 95.1% over an agent run.

### 7. Known failure modes

- DFlash2 at `CTX=huge` on this fine-tune: **do not use** (dies on prompts
  ≥ ~7k, see above).
- If the boot dies with an OOM around the split-KV verify buffer or the pool pin,
  the pool constants in the launcher were measured for a different-sized
  checkpoint — regenerate them (upstream #249) rather than guessing at `KV_MEM`.
- If decode is far below the 139.6 tok/s we measured at shallow context, check
  the power cap first (`nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv`):
  on this card a quiet 200 W box measures its cap, not the stack.

## What this repo is (and is not)

It is documentation of one box's configuration, with the measurements that box
produced. It does not fork or patch HyperQwen, it does not publish a model, and
it claims no numbers we did not measure. Anything we did not measure is labeled
unverified.

**Licenses.** The model is Apache-2.0 (UkisAI fine-tune of Qwen); HyperQwen is
Apache-2.0. This README documents both without modifying either.
