# Four-way screen, 16 October 2026

Every model began with random weights and received 2,000 updates with batch
128, seed 42, the same data stream, tokenizer and token limits. All four exact
Drive exports passed the checkpoint hash, question order and selected-token
decode checks, then were scored once by the course bot.

| Model | Peak learning rate | Validation loss | Held-out correct / 245 | Bot correct / 960 | Train + export seconds |
|---|---:|---:|---:|---:|---:|
| 9.97M, encoder 4 / decoder 4 | .001 | 1.5262 | 18 | 66 (6.88%) | 106.67 |
| 35.66M, encoder 4 / decoder 4 | .001 | 1.9631 | 12 | 31 (3.23%) | 181.02 |
| 42.48M, encoder 6 / decoder 4 | .001 | 2.0145 | 9 | 27 (2.81%) | 193.07 |
| 42.48M, encoder 6 / decoder 4 | .0003 | 1.6312 | 23 | 86 (8.96%) | 190.10 |

Retain the small model and the lower-rate 42M model. The small model has lower
token loss and lower cost; the larger finalist has higher held-out execution
agreement, confirmed by the bot. More width at rate .001 did not improve this
screen. Lowering the rate within the same 42M architecture improved all three
quality measures. This does not prove the final optimum: only one seed and a
short cosine schedule were tested, and the models remain far below the old
180k model's 53.85% public score.

Continue the two selected 2k checkpoints to 6k, retaining optimizer moments,
batch 128 and each model's peak rate. Restart a 4k cosine schedule with 200
warm-up updates. Both use the same new deterministic sample stream (seed 2042).
Save one immutable checkpoint at 6k, its predictions and numeric diagnostic.
The driver has a 900-second training timeout and 180-second diagnostic timeout
per finalist; the notebook has a 3,000-second outer timeout. No full training
is launched by this script. Check finalists on actual T4 before a long run.

The four-way driver took 718.51 seconds after data restoration. The subprocess
launcher record spans 758.79 seconds, including its restore/import work.
716 GPU samples had mean activity 72.90%, median 97%, and peak memory 18,627 MiB.
These are activity measurements, not compute efficiency. File loading, model
startup and CPU execution evaluation explain some low samples. Batch-size
throughput still needs measurement before changing the full-run batch.

All model weights and optimizer states remain in Google Drive under
HSE_NLP_HW2/runs/short-pilots-v1. The final evidence ZIP was saved there before
export, with SHA-256 recorded in export.json. Automatic release left the
runtime connected, so manual UI release was completed and verified, then CPU
was selected for file work. The bot reported current public rank 10/15 on
16 October; the preserved best score remains 53.85%. No T4 timing or private
leaderboard result is claimed for these pilots.
