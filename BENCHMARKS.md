# BENCHMARKS — task set, prefix cache, decode at depth

All numbers in this file were measured on the same box as
[README.md](README.md) / [CONFIG.md](CONFIG.md) — one RTX 3090 (24 GB), 280 W power
cap — on **2026-10-03**, and they supersede or add to the earlier measurements.
They came from the new measurement session recorded in `FINDINGS-new.md` and are
folded into the repo here. No number in this file is invented; conditions are stated
as recorded.

**Read this first: the MTP-head correction.** Every llama.cpp figure in these docs —
including the stock-llama.cpp row in the task-set table below — is a
**DISCARDED-MTP** number: stock llama.cpp silently throws away the Qwen3.8 MTP head
(15 tensors, 0.22 GiB, `-- ignoring`). The llamAmpere rows use the MTP head
(`draft-mtp`, nextn=1, n-max 4, `n_ctx_slot = 262144`, draft acceptance
0.61 / 141 accepted of 232 generated, mean len 3.43). Compare within the same MTP
state only.

## 1. Task-set comparison (5 identical tasks)

Identical 5 tasks, same box, same GGUF/family, as recorded:

| config | total |
|---|---|
| **llamAmpere v0.4 (MTP, 262k)** | **35.05 s** |
| HyperQwen fast+MTP (base Qwen3.8-27B) | 35.84 s |
| HyperQwen + Swift 1.5 (MTP) | 36.82 s |
| stock llama.cpp (**MTP discarded**) | 41.54 s |
| hand-built vLLM + MTP | 43.99 s |

Notes: the finding records the condition as "same GGUF/family" without naming the
llamAmpere GGUF's checkpoint or quantization (**unverified** in this document). The
"HyperQwen fast+MTP" row is the *base* Qwen3.8-27B, not the Swift 1.5 fine-tune; the
fine-tune on HyperQwen is the 36.82 s row. All llamAmpere numbers are taken with the
vocabulary shortlist **not** applied (section 5), so the fork's intended
configuration should be at least as fast.

## 2. Prefix cache — llamAmpere at depth

llamAmpere, MTP active, same prompt re-sent cold vs warm:

| prompt | cold | warm | speedup |
|---|---|---|---|
| 16,855 | 13.99 s | 0.15 s | 90.3x |
| 50,455 | 48.86 s | 0.19 s | 260.2x |
| 126,055 | 167.86 s | 0.21 s | 780.8x |
| 252,055 | 500.46 s | 0.36 s | **1398.8x** |

For comparison, **HyperQwen measured 256x at 252k** on the same box.

**Prompt ceiling ≠ context size.** During the same measurement session, a
262,144-token context **REJECTED a 252k-token *prompt* with HTTP 400** — the usable
prompt ceiling is below the context size. The finding records this rejection without
naming the server; the llamAmpere table above shows 252,055-token prompts being
served, so it is most plausibly the HyperQwen server (which also runs a 262,144
context). Which server exactly: **unverified**.

## 3. Decode with MTP active, at real depth (warm)

llamAmpere, MTP active, warm:

| context | prompt | warm wall | decode |
|---|---|---|---|
| ~60k | 117,665 | 3.76 s | 26.6 tok/s |
| ~120k | 243,665 | 5.49 s | 18.2 tok/s |

**WARNING — cold vs warm.** A **COLD** unique-prompt sweep of the same server shows
**2.7 tok/s at ~40k**. That number is **misleading — it is dominated by prefill, not
decode. Do not quote cold-sweep figures as decode speed.** The table above is warm
(prompt re-sent with the cache hot), which is the number that represents decode.
These are also not comparable to HyperQwen's 139.6 tok/s shallow-context figure:
different depth, different warm/cold state.

## 4. Draft acceptance (llamAmpere, MTP)

As logged by the server:

```
I slot print_timing: draft acceptance = 0.61 (141 accepted / 232 generated), mean len = 3.43
```

## 5. Open issue: the draft vocabulary shortlist is not applying

The server logs:

```
W draft_vocab_fallback: draft vocabulary shortlist not applied (no_backend_sampler:
  an output row has no backend sampler), the draft scores the full head
```

llamAmpere's own README claims **+8.45% ± 0.94%** from that shortlist. It is **not
active** on this box, so **every llamAmpere number in this repo is BELOW the fork's
intended configuration.** This is reported as an open issue, not a footnote: the
measured margin of llamAmpere over the other stacks (sections 1–3) is a
lower bound on what the fork intends to deliver. Until `draft_vocab_fallback` is
resolved, treat the +8.45% as unattained.

## 6. Why the prefix cache is the one that matters (long-horizon agent work)

An agent re-sends the accumulating history every turn, so prefix caching
(section 2) dominates turn cost at depth — not raw decode (section 3). Measured on a
real multi-file coding ticket through a real harness (the **HyperQwen** stack,
matching the original repo measurements): **39 tool calls, ~81k peak context,
11.1 minutes, 95.1% prefix-cache hit rate.** Cache is the mechanism that makes long
sessions tractable — and on that mechanism, llamAmpere at depth (1398.8x at 252k)
is dramatically ahead of HyperQwen (256x at 252k), which is why
[README.md](README.md) now recommends llamAmpere as the default for long-horizon
work while keeping HyperQwen as the stack with the verified 262k needle retrieval,
the vision tower, and the measured agent run.
