# Quality investigation — 15 October 2026

Objective: reach first place on the public leaderboard using only programs
decoded from our own randomly initialized encoder-decoder Transformer.
First place is an objective, not an expected or guaranteed outcome.

## Observations before new training

- Repository started at `29e0784`; saved model hash:
  `987a9f2b0837a718430704887d1c05bda05c0479d86d8570df14d8275b7fbaa0`.
- Corporate Colab account `a.vorsin@replika.ai`, fresh runtime verified as
  NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Public sheet inspected on 15 October: AndrejVorsin 18.85%, rank 13/15;
  first place 78.33%. The older bot receipt was rank 12/14.
- Source lengths exceed 128 tokens for only 3/1,919 test questions and
  2/1,024 existing held-out rows. Five held-out targets exceed 192 tokens.
- All 1,024 held-out programs survive byte-tokenizer encoding and decoding
  exactly with cleanup disabled. The split-identifier problem is a model
  prediction error, not a demonstrated tokenizer round-trip error.
- Samples of 1,000 rows from shards 0/15/30/45/55 have mean target lengths
  65.964/62.557/93.170/61.842/118.948. Sequential shard training may cause
  uneven learning; these observations do not establish causality.
- Shard 0 contains 24 additional training rows whose question equals a
  reserved validation question. Earlier validation excluded positions only,
  so its independence was incomplete. Future streams exclude all reserved
  question texts across all shards. Historical metrics retain this caveat.

## Provenance correction

Commit `d60fe76` removes the handwritten identifier repair. Inference retains
the exact decoded text of the greedy or syntax-selected beam candidate,
including whitespace. It saves selected token IDs and hashes and checks that
decoding those IDs reproduces every output. Existing submitted outputs and
historical scores are retained as historical evidence only.

## Planned controlled pilot

Hypothesis: mixing supplied examples across shards improves held-out numeric
agreement compared with sequential shard rotation at equal update budgets.

Use two fresh runs with identical random initialization, architecture (9.97M),
tokenizer, seed 42, precision BF16 on Blackwell, batch 64, accumulation 1,
grouping window 32, LR 0.0003, 500 warmup updates, and 6,000 total updates.
Both exclude the reserved question texts. Save latest and best every 500
updates; select on existing rows 0:255. Evaluate the saved model and each best
checkpoint on rows 256:511 using exact numeric agreement with executable
supplied references. This is a held-out diagnostic, not public pass@1.

The single experimental variable is sampling: control rotates after 1,000
mini-batches; treatment mixes a random sample of up to 4,096 original rows
from each of all 60 shards before batching. Resample each epoch. No new data,
rewritten target programs, numeric targets, or teacher outputs are used.

Compare these two pilots to each other; differences from the older run
include batch shape and held-out exclusion, so the older run is not a
single-variable control. Extend a candidate only after inspecting results.

## Remaining verification

The dataset card describes test.parquet as the project's 1,919-question test
split, but the separate pass1_questions.jsonl has not yet been obtained for
a row-by-row order comparison. Acceptance and equal counts do not prove it.
Final candidate generation must be timed on an actual NVIDIA T4 before
claiming the required 180-second limit. Blackwell timings do not establish it.

## Next bounded test, specified before running

After the matched sampling pilots, test learning rate 0.001 against 0.0003
with the mixed sampler, holding the 6,000-update budget and all other settings
fixed. This tests whether the small model is learning too slowly. Do not submit
a pilot just because token loss decreases; compare held-out numeric agreement.

## Completed sampling pilot

Control: best selection loss 1.6012 at update 4,000; held-out numeric agreement
14/245 usable references (5.71%), 46 predicted execution failures. Mixed: best
selection loss 1.4793 at update 6,000; numeric agreement 16/245 (6.53%),
42 failures. Training loop times were 177.1 and 182.6 seconds on Blackwell.
The 2-program difference is too small to establish a reliable quality gain.
Both pilots are worse than the historical saved model's 50/245 diagnostic.
No submission was sent. The learning-rate pilot is now running.

## Sampled question-order evidence

The three questions displayed by the bot after the historical submission match
official test.parquet indices 0, 1 and 2. Executing our historical programs at
those positions reproduces the values the bot attributed to our submission.
Only our reported outputs were used for this check, not the bot's reference
answers. This supports the first three positions, not the full 1,919-row order.
No program was changed or tuned from these examples.

## Completed learning-rate pilot and next budget test

At the same 6,000-update budget, LR 0.001 gives best loss 1.1409 and numeric
agreement 27/245 (11.02%), with 22 prediction execution failures. This is
better than the matched LR 0.0003 pilot (16/245), but below the old saved
model's 50/245. None of these pilots is ready for submission.

Next experiment: a fresh 20,000-update mixed run with LR 0.001, keeping every
other setting identical to the 6,000-update LR pilot. Total update budget is
the variable; cosine decay is defined over that budget. Do not describe this
as an exact continuation or as a comparison to the old 20,000-update run.
Save latest and best every 500 updates and inspect validation progress.

## Storage failure and revised workflow

The three 6,000-update pilot weights and the 20,000-update run were stored only
in temporary Colab runtime storage. The 221,498,788-byte archive was created
(SHA256 65cfd26f0f0b537972d0d6e95af16e6d8621fafccb85a22c8539d9b181c1d080),
but browser export failed with Colab output-JavaScript errors. The runtime
ended before durable export; a subsequent runtime had no quality directory.
Those pilot weights are not recovered. Their measured results remain in
quality_results.json and the actual notebook snapshot. The longer run was
last observed at update 10,400 (best logged selection loss 0.9437 at 10,000);
it did not produce a retained final checkpoint or a bot score.

The user requires CPU-only setup, a verified Drive save/export round trip
before any GPU allocation, durable results for every checkpoint, and bot
evaluation of checkpoint predictions. Future saves use immutable numbered
Drive directories, file hashes checked after copying, and completion markers.
The next 20,000-update run saves every 2,000 updates (ten checkpoints),
generates predictions from the in-memory model to avoid reloading it, and
queues every JSONL for the bot. Bot limits take precedence over the queue.
The user confirmed official test.parquet order as the required task order;
the separate pass1_questions.jsonl is no longer a prerequisite.

## CPU Drive verification, 15 October 2026

The approved a.vorsin@replika.ai Drive mount passed a real browser export:
setup_verified.txt SHA256 79bc3c134a038e7c283ef8c2dd7371171096ec7c657c8d8a822192b643a7d842
matched locally. All eight custom checks passed on CPU. All 60 training shards
and official test.parquet (1,919 questions), source bundle, historical model
and six evidence files were copied to Drive. A restore into empty directories
verified all 61 Parquet hashes and the historical model hash. The final
setup-complete.json was also downloaded through Drive and checked locally.
See submissions/2026-10-15-drive-setup.json. No GPU was allocated for these checks.
Training remains paused; this storage check is not a new bot submission.


## Completed Drive run and longer continuation, 15 October 2026

The mixed-lr001-20k-drive-v1 run completed and all ten saved checkpoints
were exported from Drive and evaluated by the bot. The best public score
was 20.31% (195/960) at update 20,000; the observed rank was 13/15.
Selection loss was still improving (0.79525 at 20,000). See the run summary
and ten receipts in submissions/. Compute was disconnected afterward.

At the user's request, the next run continues our own verified 20,000-update
checkpoint with optimizer moments retained and a fresh warmup/decay phase.
Before choosing its batch size, benchmark 64, 128, 256 and 512 on identical
training examples, in forward and reverse order. Record examples/second,
tokens/second, CPU preparation time, GPU compute time, peak memory and
nvidia-smi samples. This hardware probe does not select a trained candidate;
its temporary weights are discarded. Prepare its inputs and source on CPU
and save them to Drive before allocating GPU. Every continuation checkpoint
and its exact test outputs must again be saved to Drive and scored by the bot.


## Batch throughput result and completed continuation

Both passes measured 16,384 identical examples (1,210,118 non-padding target
tokens per pass). Median examples/second: batch 64 = 2,100.34; 128 = 2,278.92;
256 = 2,114.27; 512 = 1,949.52. Batch 128 was selected: 8.5% more examples
per second than 64, with peak allocated memory 7.75 GiB. The synchronized
probe's average GPU utilization rose from 45% at batch 64 to 60% at 512,
even though useful throughput fell at 512. The real unsynchronized training
loop subsequently showed instantaneous utilization of 69-83%; these samples
are not a run average. Raw telemetry is retained in Drive.

continue-20k-60k-b128-v1 started from our retained 20,000-update model and
optimizer. It adds 40,000 updates at batch 128, peak LR 0.0005, 500-update
warmup and cosine decay. Checkpoints and test outputs are saved every 5,000
updates. Source a411a64f8afe61e3a2b337af338c0ea6c19c9c15 is pinned. The new
notebook documents CPU preparation, GPU restore, benchmark and continuation.
The run completed all 60,000 updates in 1,957.2 seconds (32.6 minutes for the
continuation, including periodic evaluation and saves). It presented another
5,120,000 training examples. All eight new checkpoints were saved to Drive,
exported from Drive, verified, and scored by the bot.

| Step | Public correct / 960 | Score |
| --- | --- | --- |
| 25,000 | 194 | 20.21% |
| 30,000 | 227 | 23.65% |
| 35,000 | 231 | 24.06% |
| 40,000 | 273 | 28.44% |
| 45,000 | 314 | 32.71% |
| 50,000 | 343 | 35.73% |
| 55,000 | 353 | 36.77% |
| 60,000 | 362 | 37.71% |

The best result is 37.71% at step 60,000, rank 12 of 15 at 19:13 +03:00.
The 33% baseline is exceeded; first place remains unmet. GPU telemetry
contains 1,972 one-second samples: mean utilization 60.38%, median 78%,
16.23% zero-utilization samples, and peak total used memory 10,397 MiB.
This includes saving and generation, so it is not a training-only average.
CPU preparation occupied about 41% of the matched batch-128 probe.
Colab showed no active sessions after release at 19:07 +03:00.

See the run summary, raw telemetry, launch settings, and eight receipts.
Fresh full-generation T4 timing and held-out numeric evaluation remain open.

## Planned input pipeline probe and 60k-to-180k continuation

The 60k model still improved in both selection loss and public score at each
of the last six saved checkpoints. The next quality experiment adds 120,000
updates to the same model, with the same tokenizer and allowed-data sampler.
It retains AdamW moments, uses batch 128, BF16, and a fresh 500-update warmup
with peak rate 0.0003 followed by cosine decay. Six checkpoints at 80k, 100k,
120k, 140k, 160k and 180k will each be retained in Drive and scored by the bot.
No architecture, target text, outside examples or handwritten answer changes
are introduced. This run tests whether additional training improves quality;
its new schedule is explicitly part of the continuation.

Before GPU allocation, compare every tensor from three input paths on 32,768
examples from the existing official training-only benchmark pool: legacy;
window tokenization reused for collation; and that path with a bounded
2-batch CPU preparation queue. Require identical shapes, IDs, attention masks,
labels and order. Local tests cover truncation, duplicate rows, partial windows,
literal special tokens, producer errors and early consumer shutdown (18 tests
passed). Synthetic fixtures are confined to tests and never enter training.

Then run two reversed-order GPU probes using the same checkpoint, seed,
examples, batch size, precision, optimizer and warmup. Measure whole-loop
examples/second without synchronizing each update. Keep the legacy path unless
the fastest candidate improves median throughput by at least 5%. Probe weights
are discarded. This changes only how the same batches are prepared.

The CPU stage verifies all cached Parquet files and the 60k checkpoint, reopens
the model, saves source and equivalence evidence to Drive, and copies the prior
run's eight bot receipts and final report to Drive. The GPU stage reuses these
verified inputs, retains telemetry, and requests runtime release at completion
or failure. Runtime release must also be checked in Colab's active sessions UI.

### Verified probe result and continuation launch

The CPU gate passed on source `ca26869455692cb570b94cb4a633be7701221f9e`:
all three paths produced identical tensor hashes for 32,768 examples and
2,419,826 target tokens. CPU preparation took 18.72 seconds for legacy,
6.38 for reused tokenization, and 5.98 with the two-batch queue. All eight
prior bot receipts, summaries, the report and inference notebook were copied
to the prior Drive run folder and verified by SHA256 before GPU allocation.

On the RTX PRO 6000 Blackwell, median throughput over reversed rounds was:

| Input path | Examples per second |
| --- | ---: |
| Legacy | 2,931.17 |
| Reused tokenization | 3,483.24 |
| Reused tokenization plus CPU queue | 3,692.80 |

The selected path improves throughput by 25.98% in this matched probe, with
the same batch size of 128 and peak allocated tensor memory of 7.75 GiB.
The 60k-to-180k continuation was launched with this selected path. This is
a speed result; new validation losses and bot scores remain pending.
See `experiments/continue-60k-180k-b128-v1/` for CPU equivalence, full probe
records, selection, source hashes and the launch configuration exported from
Drive. Training completion, all six bot scores and GPU release must be
verified before reporting this continuation as complete.

The first continuation checkpoint, step 80,000, achieved validation token loss
0.525003 (previous 60k: 0.565194) and bot public pass@1 **43.13% (414/960)**,
rank **12 of 15** at 20:00 +03:00 on 2026-10-15. This improves the previous
best by 52 correct questions. All 1,919 outputs were saved to Drive first,
exported with a verified archive hash, and checked against both the CSV and
the exact decoded model token sequences before submission. Its receipt was
written and read back in the Drive run folder, then exported to the repository.
Generation took 9.585 seconds on Blackwell; actual T4 timing remains open.
The same training process continued past step 84,200; later checkpoints and
their bot scores are still pending. First place has not been achieved.

Step 100,000 achieved validation token loss 0.490893 and bot public pass@1
**47.40% (455/960)**, rank **11 of 15** at 20:13 +03:00 on 2026-10-15.
This adds 41 correct questions over the 80k checkpoint. All 1,919 outputs
were saved to Drive before export; archive, file hashes, question order,
checkpoint provenance and exact token decoding were verified before the
single submission. The score receipt was written and read back on Drive.
Generation took 8.646 seconds on Blackwell; this does not verify T4 timing.
Training is continuing toward 180k with the same process and batch size 128.
