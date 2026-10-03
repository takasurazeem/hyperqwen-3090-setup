# CONFIG — exact values and measured benchmarks (both stacks)

Reference: [README.md](README.md), [BENCHMARKS.md](BENCHMARKS.md). Everything in this
file is either (a) a value this box actually runs, (b) a measurement this box actually
produced, or (c) explicitly marked **unverified**. Two stacks are documented:

- **Stack A — HyperQwen** (patched vLLM), the original setup of this repo.
- **Stack B — llamAmpere** (llama.cpp fork, MTP head active), added with the
  2026-10-03 measurements.

## Stack A — HyperQwen (patched vLLM)

### Environment values

The serving environment, as run on the box:

| Variable | Value | Meaning |
|---|---|---|
| `CTX` | `huge` | HyperQwen context profile: KVarN 4/2-bit KV backend, the only profile that holds 262k on 24 GB |
| `MAX_LEN` | `262144` | `--max-model-len`; the model's native context |
| `SPEC` | `mtp` | speculative drafter = Qwen's own MTP head |
| `DRAFT_TOKENS` | `3` | drafts proposed per MTP pass |
| `PREFIX_CACHE` | `1` | vLLM prefix caching on (recurrent-state resume included) |
| `VISION` | `1` | vision tower served (CPU-offloaded in the stock image) |
| Power cap | `280 W` | set with `sudo nvidia-smi -pl 280` (upstream's reference card is capped at 250 W) |
| Served port | `18020` | OpenAI-compatible API |

Everything else in the launcher runs at its upstream default
(`GPU_UTIL`, `KV_MEM`, `INT8_ACT`, `MAX_SEQS`, …). If you override any of those,
you are no longer on this configuration.

### Model

| | |
|---|---|
| Checkpoint | `ukisai/Swift-1.5-Qwen3.8-27b` (Hugging Face; not on GitHub) |
| What it is | SWIFT-1.5 fine-tune of Qwen3.8-27B, **AWQ INT4** (W4A16) |
| Architecture id | `qwen3.5` |
| Layers | 48 linear-attention (Gated DeltaNet) + 16 full-attention blocks |
| KV heads | 4 (head_dim 256) |
| Vocab | 248,320 |
| Native context | 262,144 tokens |
| Heads at serve time | `lm_head` + `embed_tokens` + MTP module **requantized bf16 → int8** by HyperQwen's `prepare/` pipeline (the crux; see README) |

### Server and hardware

| | |
|---|---|
| Server | [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) — patched vLLM (upstream's badge says vLLM 0.29.0; **the exact commit/image our box runs: unverified in this document**) |
| Container | the upstream Docker image (9.5 GB), `--profile single` |
| GPU | one RTX 3090, 24 GB, sm86 |
| Power | 280 W cap |
| OS | Linux (distro of our box: **unverified in this document**) |

## Stack B — llamAmpere (llama.cpp fork)

### Build and serving values

As built and run on the box (2026-10-03):

| Item | Value | Meaning |
|---|---|---|
| Fork | [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) **v0.4** (MIT) | llama.cpp fork for Ampere (RTX 3090 / 3090 Ti) |
| MTP head | **used** | stock llama.cpp discards it (15 tensors, 0.22 GiB, `-- ignoring`) |
| Drafter | MTP, **on by default for `qwen35`**: `draft-mtp`, nextn=1, n-max 4 | Qwen's own MTP head as speculative drafter |
| Draft vocabulary shortlist | requested as `'auto'`; **not applied** on this box | open issue `draft_vocab_fallback` — see BENCHMARKS.md |
| Slots | `n_slots = 1` | single slot |
| Context slot | `n_ctx_slot = 262144` | the model's native context |
| CUDA build | CUDA 13.2, sm_86, host compiler **gcc-15** | `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` (GCC 16 is rejected by nvcc) |
| Model file | GGUF, same family as the Stack A checkpoint | **exact quantization: unverified in this document** |
| GPU / power | one RTX 3090, 24 GB, 280 W cap | same box as Stack A |
| API | llama.cpp server (OpenAI-compatible) | **port and agent-run behaviour: unverified in this document** |

## Measured benchmarks

### Stack A — HyperQwen

All four numbers were measured on the box described above, with the environment
values in the table.

| # | Measurement | Value | Conditions |
|---|---|---|---|
| A1 | Decode rate, shallow context | **139.6 tok/s** | `CTX=huge` profile, shallow prompt (depth of "shallow": **unverified** — this is the box's short-context decode rate) |
| A2 | 262k context, needle retrieval | **retrieved** | needle planted at a **207k-token prompt**; the model quoted it back |
| A3 | Prefix-cache hit rate | **95.1%** | over a real agent run (repeated-conversation traffic), `PREFIX_CACHE=1` — 39 tool calls, ~81k peak context |
| A4 | Real multi-file coding ticket | **11.1 minutes** end to end | one genuine multi-file coding task through the OpenAI-compatible API, including tool calls |

### Stack B — llamAmpere

Measured on the same box, 2026-10-03. All with the MTP head **active** (and the
vocabulary shortlist **not** applied — see BENCHMARKS.md, open issue).

| # | Measurement | Value | Conditions |
|---|---|---|---|
| B1 | Task set, 5 identical tasks | **35.05 s** total | same box, same GGUF/family; best of the five configs in BENCHMARKS.md |
| B2 | Prefix cache, 16,855-token prompt | 13.99 s cold → **0.15 s** warm (90.3x) | warm = prompt re-sent with cache hot |
| B3 | Prefix cache, 50,455-token prompt | 48.86 s cold → **0.19 s** warm (260.2x) | as above |
| B4 | Prefix cache, 126,055-token prompt | 167.86 s cold → **0.21 s** warm (780.8x) | as above |
| B5 | Prefix cache, 252,055-token prompt | 500.46 s cold → **0.36 s** warm (**1398.8x**) | as above |
| B6 | Decode at real depth (warm), ~60k context | **26.6 tok/s** | 117,665-token prompt, 3.76 s warm wall, MTP active |
| B7 | Decode at real depth (warm), ~120k context | **18.2 tok/s** | 243,665-token prompt, 5.49 s warm wall, MTP active |
| B8 | Draft acceptance | **0.61** (141 accepted / 232 generated), mean len 3.43 | MTP drafter, as logged by the server |
| B9 | Cold unique-prompt sweep at ~40k | 2.7 tok/s | **NOT a decode number** — dominated by prefill; do not quote as decode speed |

### Cross-stack comparisons (same box, 2026-10-03)

| # | Measurement | Value | Conditions |
|---|---|---|---|
| C1 | Task set — HyperQwen fast+MTP, **base** Qwen3.8-27B | 35.84 s | same 5 tasks as B1 |
| C2 | Task set — HyperQwen + Swift 1.5 (MTP) | 36.82 s | same 5 tasks as B1; this is the fine-tune on Stack A |
| C3 | Task set — stock llama.cpp (**MTP discarded**) | 41.54 s | same 5 tasks as B1; the discarded-MTP reference |
| C4 | Task set — hand-built vLLM + MTP | 43.99 s | same 5 tasks as B1 |
| C5 | Prefix cache at 252k — HyperQwen | **256x** | comparison point for B5's 1398.8x |
| C6 | Prompt ceiling | a 262,144-token context **REJECTED a 252k-token prompt with HTTP 400** | usable prompt ceiling is below the context size; the finding does not name the server (llamAmpere served 252,055-token prompts, so most plausibly HyperQwen) — **which server: unverified** |

## Drafter selection (why MTP, not DFlash2)

| Drafter | At `CTX=huge` on this fine-tune |
|---|---|
| `SPEC=mtp` (Stack A) | stable; the 95.1% prefix-cache hit rate above is this drafter's |
| `SPEC=dflash2` | **unstable**: dies on prompts ≥ ~7k tokens. Not used. |
| MTP head, llamAmpere (Stack B) | on by default for `qwen35`; 0.61 draft acceptance measured on this box |

The ~7k figure is a threshold observed on this box, not a spec of DFlash2.

## Unverified in this document

We did not measure or record the following; do not treat them as facts about this
setup:

**Stack A (HyperQwen)**

- the exact HyperQwen commit / image digest the box is pinned to
- the KV-pool size in tokens at boot (the launcher prints it; we did not record it here)
- prefill/TTFT numbers at 262k (only the needle retrieval above was run there)
- power draw and clock behaviour under sustained load
- the exact prepare-script path used for *our* checkpoint (symmetric vs
  asymmetric AWQ body determines the script; see README step 3)
- the distro/driver versions on the box

**Stack B (llamAmpere)**

- the exact GGUF quantization used in the measurements (the finding says "same
  GGUF/family" without naming the quant)
- the exact llama.cpp base commit / llamAmpere build the box ran
- which server returned the HTTP 400 for the 252k prompt (C6)
- vision support on llamAmpere (no measurement)
- a needle-retrieval test on llamAmpere (no measurement)
- an agent-run (tool-call) measurement on llamAmpere (the 11.1-minute / 95.1% ticket
  is the Stack A number)

## Related measurements that are NOT ours (for scale)

Upstream HyperQwen, same card family (250 W, *stock* model, DFlash2): 127 tok/s
single-stream, ~1,035 tok/s at 64 concurrent; 262k context via KVarN in the
`CTX=huge` profile. Their
[`docs/reproductions/`](https://github.com/syv-ai/HyperQwen/tree/main/docs/reproductions)
table is the place to compare hardware, but note the power cap and model differ
from ours on every row.
