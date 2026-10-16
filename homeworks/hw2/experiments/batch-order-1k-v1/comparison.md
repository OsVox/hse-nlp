# Grouped batch order: short paired experiment

Four 1,024-update continuations from the same own 70k model and optimizer. Each saw 131,072 original pairs at batch 128. Within each seed, only the order of completed length-grouped batches changed. Original examples, batch membership, padding cost and targets were preserved. Source: `862fa6f491c6c1bd4c7977d688aa37deaf2c54db`.

| Seed / order | Pilot /245 | Confirmation /493 | Execution failures | Public /960 |
|---|---:|---:|---:|---:|
| 42, control | 137 | 281 | 41 | 561 (58.44%) |
| 42, shuffled | 136 | 285 | 45 | 550 (57.29%) |
| 1042, control | 138 | 284 | 49 | 560 (58.33%) |
| 1042, shuffled | 135 | 286 | 48 | 549 (57.19%) |

The predefined held-out gate failed. Shuffling gained 3 correct answers for seed 42 and lost 1 for seed 1042; both pilot subsets declined and pooled execution failures increased by 3. Public scores independently fell by 11 correct answers in each matched pair. No long extension or combination is justified by this experiment. These reused held-out questions are exploratory, not independent replicated test sets.

All four checkpoints and their original greedy outputs were saved to Drive before export. Exact code strings, 1,919-row official question order, model/source hashes and no-repair decoding were verified. Each unique JSONL was scored once by the bot; receipts were uploaded and read back from Drive. T4 timing remains unverified by user choice.

Driver duration was 516.16 seconds (8.60 minutes), excluding runtime allocation and OAuth beforehand. Median sampled GPU utilization was 97%; mean was 63.79%, including restoration, CPU evaluation and saving. Colab showed no active sessions after automatic release. Fifteen focused tests passed before launch. The best retained public result remains 573/960 (59.69%), rank 8/15; first place is not achieved.

Next research direction: test number segmentation with an unchanged vocabulary in a short matched adaptation experiment. First measure exact decoding and token-length effects on CPU. This is a hypothesis, not a demonstrated improvement. Also consider overlapping CPU scoring with the next GPU training arm to reduce idle compute without changing evaluation.
