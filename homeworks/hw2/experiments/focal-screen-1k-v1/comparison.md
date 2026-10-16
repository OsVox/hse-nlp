# Focal loss: three short trials

Completed 2026-10-16. Each arm continued the same own 70,000-step checkpoint and Adam state for 1,024 updates, batch 128, seed 42, peak learning rate 0.00003 and 50 warmup updates. Original data, programs, vocabulary and strict greedy decoding were unchanged. Source: `86c3bfb603eeb2f20c7e3b4cf7dd3f246b084d93`.

| Loss | Pilot /245 | Confirmation /493 | Total /738 | Execution failures | Bot /960 |
|---|---:|---:|---:|---:|---:|
| Control | 137 | 281 | 418 | 41 | 561 (58.44%) |
| Focal gamma 1 | 139 | 283 | 422 | 40 | 559 (58.23%) |
| Focal gamma 2 | 138 | 278 | 416 | 42 | 561 (58.44%) |

Gamma 1 passes the predefined exploratory gate: improvement on both held-out subsets and no added total execution failures. It is only a candidate for a short second-seed confirmation, not a long training run. Gamma 2 fails the gate. Neither improves the public score. The retained best remains numeric2 at 573/960 (59.69%); the bot reports rank 8/15.

Focal loss multiplies each token's cross-entropy by `(1-p)^gamma`; the gradient includes the multiplier. It changes gradient magnitude as well as relative emphasis, so matching the nominal learning rate does not isolate emphasis alone. Repeated use of the held-out questions makes these comparisons exploratory. Public scores are recorded separately from the selection gate, and public failure examples do not supply training targets or repairs.

## Compute and scoring

The complete driver took 381.29 seconds (6.35 minutes), excluding prior allocation and authorization. Median sampled GPU activity was 97%; mean was 65.68%, including restore, saving and evaluation. Peak sampled memory was 21,077 MiB. No long run started. Colab's Manage Sessions dialog confirmed no active sessions after automatic release; see `gpu-released.png`.

Evaluation now saves generated programs and original references before a separate CPU-only process scores them. This permits the next GPU task to start without waiting for scoring. Hashes and row order are checked, failures propagate, and the scoring branch imports no torch or model weights. Actual scoring intervals contain 34 one-second GPU samples: only 3 show nonzero GPU activity and none reaches 90%. Thus overlap occurred, but most scoring overlapped setup rather than heavy GPU computation. There is no controlled evidence of a wall-time speedup yet.

Twenty focused checks passed before launch, covering focal gradients, padding, preserved scoring results, tamper detection and subprocess failures. Previously CPU-verified data and parent archives were reused, with archive hashes rechecked at restore. This avoided another redundant Colab CPU preflight.

## Artifacts

Every checkpoint and original 1,919-row test output was saved to Drive before export. All three unique outputs were verified against selected token IDs, model/source hashes, greedy settings and official test question order before one bot submission each. Bot receipts were uploaded to their checkpoint folders and read back byte-for-byte.

- [Drive experiment folder](https://drive.google.com/drive/folders/193ciYupvFfG0asWQZBDVpruQ22WRPZh2)
- `plan.json`: preserved preregistration; its prepared status is historical.
- `runtime-manifest.json`, `execution.json`, `selection.json`: executed configuration and outcomes.
- `utilization-summary.json`, `scoring-execution.json`: resource and scoring evidence.
- Per-arm folders contain held-out diagnostics, exact exports, verification and bot receipts.

T4 timing remains unverified and deferred. No claim of first place or a proven generalization improvement is made.
