# Focal loss: second-seed confirmation

Completed 2026-10-16. Two new matched 1,024-update continuations use seed 1042, the same own 70k parent checkpoint and Adam state, batch 128, original tokenizer/data/programs, peak LR 0.00003 and 50 warmup updates. Only gamma-one focal loss differs from control. The prior seed's completed results are reused without retraining. Source: `bbf58a9376348573d28a72b798113fa0d6bab6a9`.

| Seed / loss | Pilot /245 | Confirmation /493 | Total /738 | Execution failures | Bot /960 |
|---|---:|---:|---:|---:|---:|
| 42 / control | 137 | 281 | 418 | 41 | 561 |
| 42 / focal1 | 139 | 283 | 422 | 40 | 559 |
| 1042 / control | 138 | 284 | 422 | 47 | 560 |
| 1042 / focal1 | 138 | 286 | 424 | 46 | 563 |

The second seed gains two held-out agreements and removes one execution failure, but the pilot subset ties control. Therefore the preregistered rule requiring positive gains on both subsets in each seed fails. No combination or long extension is justified by that gate. Public effects differ by seed: -2 and +3 correct answers. These small mixed changes do not establish a useful public improvement. The best checkpoint remains numeric2 at 573/960 (59.69%); latest bot rank is 8/15.

The repeated held-out questions are exploratory diagnostics, not independent test samples. Focal weighting changes gradient magnitude as well as token emphasis. The first-seed results are bundled with the source and their SHA-256 is checked against the recorded prior evidence. No public failure examples supply targets or repairs.

## Compute and verification

The driver took 269.42 seconds (4.49 minutes), excluding allocation and Drive connection. GPU median utilization was 97%, mean 64.09%, including restore, generation, scoring and saves; peak sampled memory was 21,309 MiB. Batch size was deliberately held fixed to preserve the comparison. Both runs completed and Colab Manage Sessions showed no active sessions after automatic release.

Twenty-one focused checks passed. Unchanged data and parent archives reused the earlier CPU preflight; new source, launcher and metadata were uploaded to Drive and read back before GPU allocation. Restoration verifies archive hashes. Both checkpoints and their 1,919 original greedy outputs were saved to Drive first, exported, and verified against selected token IDs, CSV strings, question order and source/model hashes before one bot submission per unique output. Receipts are saved in checkpoint folders and read back for verification.

[Drive experiment folder](https://drive.google.com/drive/folders/1Ka0A972z6udqDIjbE5S26wcf027YqEkO). See `selection.json`, `execution.json`, `utilization-summary.json`, `runtime-release.json` and per-arm export/receipt evidence. `plan.json` preserves preregistration; `runtime-manifest.json` records the executed preparation. T4 timing remains unverified and deferred.

The next distinct hypothesis is a CPU audit of agreement among original programs attached to identical training questions. This is only an audit proposal: disagreements do not identify the correct answer, and unsupported execution does not prove a bad target. No filtering or new targets have been applied.
