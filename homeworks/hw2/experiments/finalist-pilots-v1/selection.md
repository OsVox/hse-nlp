# Finalist comparison, 16 October 2026

Both models continued their own 2,000-step checkpoint, including Adam moments,
to 6,000 updates. Batch 128, BF16, original training targets, the same new
deterministic mixed stream, and a restarted 4k cosine schedule with 200 warm-up
updates were held fixed. Source: 10db3cda45ba777e04e4195f003a6b1db5254b24.

| Model | Peak rate | Validation loss | Held-out correct /245 | Bot correct /960 | Added training + export seconds |
|---|---:|---:|---:|---:|---:|
| 9.97M, encoder 4 / decoder 4 | .001 | .946274 | 38 | 154 (16.04%) | 183.73 |
| 42.48M, encoder 6 / decoder 4 | .0003 | .978169 | 41 | 182 (18.96%) | 353.36 |

The larger finalist leads execution accuracy in both evaluations, while the
small model has lower next-token loss and costs less. The public difference
is 28 answers (2.92 percentage points); the held-out difference is only three
answers. These are one-seed, early-learning comparisons, not proof of an
optimal size or statistical significance. Both are below the established
180k small model's 53.85%. The best public rank remains 10/15 at this snapshot.

The added training loop took 160.2 seconds for the small model and 333.3 for
the larger one; the broader per-trial times above include startup, validation,
checkpoint saves and all-test generation. At the measured larger-model rate,
one corpus-sized amount of examples would take roughly 2.1 hours of training
loop time at batch 128, before extra evaluation and archival. This revises the
older parameter-linear 3.8–4.4-hour planning estimate, but is still an
extrapolation from a short run rather than a completed full-corpus measurement.

Across 583 GPU samples, median activity was 96%, mean 72.15%, and peak used
memory was 18,631 MiB. These activity figures include I/O and evaluation; they
do not establish compute efficiency or the best batch size.

Both immutable checkpoints, optimizer states, exact predictions and bot
receipts are saved in Drive. The evidence archive was created on Drive before
export; its hash is in export.json. Every selected token sequence was decoded
locally and checked against the submitted JSONL. No answer text was edited.
The completed G4 runtime was manually released after the asynchronous release
request left it connected. Standard CPU was used for subsequent preparation.

Next: actual T4 inference for the established model and larger finalist, then
a bounded production-pipeline training batch probe. A faster training batch
must also pass a short equal-example quality comparison before full training.
The 42M model is a candidate for the next stage, not a demonstrated optimum.
