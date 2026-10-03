# CONFIG — exact values and measured benchmarks

Reference: [README.md](README.md). Everything in this file is either (a) a value
this box actually runs, or (b) a measurement this box actually produced, or
(c) explicitly marked **unverified**.

## Environment values

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

## Model

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

## Server and hardware

| | |
|---|---|
| Server | [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) — patched vLLM (upstream's badge says vLLM 0.29.0; **the exact commit/image our box runs: unverified in this document**) |
| Container | the upstream Docker image (9.5 GB), `--profile single` |
| GPU | one RTX 3090, 24 GB, sm86 |
| Power | 280 W cap |
| OS | Linux (distro of our box: **unverified in this document**) |

## Measured benchmarks

All four numbers were measured on the box described above, with the environment
values in the table. No other numbers in this repository are ours.

| # | Measurement | Value | Conditions |
|---|---|---|---|
| 1 | Decode rate, shallow context | **139.6 tok/s** | `CTX=huge` profile, shallow prompt (depth of "shallow": **unverified** — this is the box's short-context decode rate) |
| 2 | 262k context, needle retrieval | **retrieved** | needle planted at a **207k-token prompt**; the model quoted it back |
| 3 | Prefix-cache hit rate | **95.1%** | over a real agent run (repeated-conversation traffic), `PREFIX_CACHE=1` |
| 4 | Real multi-file coding ticket | **11.1 minutes** end to end | one genuine multi-file coding task through the OpenAI-compatible API, including tool calls |

### Drafter selection (why MTP, not DFlash2)

| Drafter | At `CTX=huge` on this fine-tune |
|---|---|
| `SPEC=mtp` (ours) | stable; the 95.1% prefix-cache hit rate above is this drafter's |
| `SPEC=dflash2` | **unstable**: dies on prompts ≥ ~7k tokens. Not used. |

The ~7k figure is a threshold observed on this box, not a spec of DFlash2.

## Unverified in this document

We did not measure or record the following; do not treat them as facts about this
setup:

- the exact HyperQwen commit / image digest the box is pinned to
- the KV-pool size in tokens at boot (the launcher prints it; we did not record it here)
- prefill/TTFT numbers at 262k (only the needle retrieval above was run there)
- power draw and clock behaviour under sustained load
- the exact prepare-script path used for *our* checkpoint (symmetric vs
  asymmetric AWQ body determines the script; see README step 3)
- the distro/driver versions on the box

## Related measurements that are NOT ours (for scale)

Upstream HyperQwen, same card family (250 W, *stock* model, DFlash2): 127 tok/s
single-stream, ~1,035 tok/s at 64 concurrent; 262k context via KVarN in the
`CTX=huge` profile. Their
[`docs/reproductions/`](https://github.com/syv-ai/HyperQwen/tree/main/docs/reproductions)
table is the place to compare hardware, but note the power cap and model differ
from ours on every row.
