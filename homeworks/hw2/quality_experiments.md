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

Step 120,000 achieved validation token loss 0.471192 and bot public pass@1
**49.17% (472/960)**, rank **10 of 15** at 20:26 +03:00 on 2026-10-15.
This adds 17 correct questions over step 100k. All 1,919 outputs passed
the same Drive-first archive, manifest, question-order and exact-token-decode
checks before one bot submission. The score receipt was saved and read back
on Drive. Blackwell generation took 10.212 seconds; actual T4 timing is still
unverified. The live process continued beyond 120k; 140k–180k remain pending.

### Planned held-out decoding comparison at step 120k

Hypothesis: four-beam decoding improves numeric agreement with the supplied
held-out training references enough to justify its extra inference cost.
Export the unchanged, verified step-120000 checkpoint from Drive. Compare
greedy and four-beam generation on the same reserved shard-0 rows 256–511,
using FP32, batch 64, input limit 128 and output limit 192. Use the existing
numeric evaluator on the local CPU, leaving Colab training uninterrupted.
Record model/source/data hashes and both full-generation wall times. Treat
these CPU times only as relative diagnostic costs, never as T4 evidence.
Do not use public-test answers or bot examples, modify programs, or create
training targets. Keep the current submission decoder unless held-out
evidence supports a change and the full T4 timing gate later passes.

Result: greedy decoding matched 120/245 usable references (48.98%), while
four beams matched 125/245 (51.02%). Prediction execution failures fell from
19 to 15. Generation took 10.512 and 57.190 seconds respectively on the same
local CPU with four threads (5.44 times as long). Both runs used the same
model and question hashes; exact generated programs and evaluator hashes
are retained in `experiments/decoding-120k/`. This small single-set gain is
insufficient to change the submission decoder, especially while T4 timing
for four beams remains unknown. Retain greedy decoding with the existing
model-only retry for syntactically invalid outputs. These are held-out
training diagnostics, not public scores or evidence of T4 compliance.

Step 140,000 achieved validation token loss 0.452829 and bot public pass@1
**50.73% (487/960)**, rank **10 of 15** at 20:39 +03:00 on 2026-10-15.
This adds 15 correct questions over step 120k. Its 1,919 outputs were saved
to Drive first, exported and checked against the manifest, official order
and exact selected model token decodes before one bot submission. The
receipt was written and read back on Drive. Blackwell generation took
9.372 seconds; T4 timing remains open. Training continues toward 180k.

### Planned checkpoint-average pilot (CPU only)

Compare the unchanged step-140000 model with the equal-weight average of
steps 120000 and 140000 from the same run. Verify both parent hashes and
identical architecture/tokenizer/run settings, average floating tensors in
CPU float64, cast back to their original dtype, and reopen the saved T5.
Keep both parents immutable and record all parent hashes. Use the same
reserved training rows 256–511, FP32, greedy decoding, batch 64 and length
limits as the preceding decoding comparison. Select on held-out numeric
agreement and validation token loss; do not use test answers or bot examples.
The average introduces neither new targets nor inference-time model copies.
It is a pilot, not a claim of improvement; later averages need fresh checks.

Result: the average matched 124/245 usable references, versus 122/245 for
the unchanged 140k model. Execution failures increased from 16 to 22, while
selection token loss worsened from 0.452825 to 0.459383 on the same CPU and
rows 0–255. Generation times were 10.160 and 10.313 seconds. This is mixed
evidence on a small set, so retain the unchanged checkpoint. Exact predictions,
model hashes and both diagnostics are in `experiments/averaging-120k-140k/`.

Step 160,000 achieved validation token loss 0.441922 and public bot pass@1
**53.75% (516/960)**, rank **9 of 15**, at 20:54 +03:00 on 2026-10-15.
This adds 29 correct questions over 140k. The Drive-first export and exact
token decoding checks passed; the single submission's receipt was saved and
read back on Drive. Blackwell generation took 9.495 seconds. The improvement
is larger than the previous interval, so a hard quality plateau is not proven.

### Planned T4 inference batch-size comparison

At the user's request, prioritize inference speed before another training run.
Use the immutable 160k checkpoint, FP32, current greedy decoder and syntax-only
four-beam retry. Compare batch sizes 64, 128 and 256, twice in reversed order,
with input/output limits 128/192. Retain every exact program, its model/token
provenance, peak allocated memory and output differences from batch 64. Only
unchanged outputs can establish a speed gain without a new quality comparison.
Profile a separate baseline pass so its synchronization overhead cannot alter
the primary timing comparison. Measure model loading and warm-up separately;
include all generation, retry, decode, local CSV and full-call overhead in
reported scopes. Save results to Drive after each run. Bound the experiment
to a 600-second soft budget and 720-second process timeout, then release GPU.
All model/data loading checks must pass on CPU before allocating the T4.
Do not benchmark concurrently with training. No performance result yet.

The corpus has 11,838,003 rows in 60 shards. The existing train-only probe
contains 15,358 rows and estimates 46.91 capped question tokens and 73.83
capped code tokens per pair: about 555M source and 874M target tokens per
corpus. The probe samples near-equal numbers per shard; these are approximate
totals, not unique-content counts. The Chinchilla D/20 heuristic gives 44M
using targets or 71M using combined tokens, but neither is a fitted optimum
for our encoder-decoder. See `scaling_estimate.py` and
`experiments/scaling-analysis/data-estimate.json`. Proposed first larger
candidate: width 512, six encoder layers, four decoder layers, 42,481,152
parameters. Its quality and T4 speed must be measured before long training.

The continuation finished at 180,000 with return code 0 and all six checkpoint
exports complete on Drive. Its last checkpoint achieved token loss 0.440206
and **53.85% (517/960), rank 9/15** at 21:14 +03:00. This is one more correct
answer than 160k; no further continuation was started. Generation took 8.800
seconds on Blackwell. The GPU was released through Colab controls and file
work moved to a verified CPU runtime. Colab's JavaScript download failed for
the final archive; the saved Drive archive was exported through Colab MCP in
bounded byte chunks and verified against its SHA-256 before submission.

The planned T4 probe now uses the final 180k checkpoint (model SHA-256
0399a924eea6ef872aa0b403b77fc5e33d24909cc7690536880bc00b254362c7),
with the same predeclared decoder and batch-size comparison. This also gives
an actual T4 timing gate for the final trained checkpoint. No T4 result yet.

CPU preparation passed for the 180k checkpoint: model loading and a short
generation check succeeded, and the hashed input bundle plus benchmark notebook
were saved and verified on Drive. The first T4 attempt failed at `drive.mount`
before generation. After releasing the GPU, a browser refresh and CPU-only
retry failed again. DriveFS reported that its internal credential request failed;
the browser also reported output-JavaScript and proxy-frame loading errors.
This does not establish a missing Drive permission or an inference-code fault.
The GPU remains off. No T4 timing, batch-size winner, or speedup can be reported.
Evidence is in `experiments/inference-speed-180k-t4-v1/`; the latest failure
records and report are preserved locally and in Git, pending Drive recovery.


## 16 October: short size/rate pilots and actual T4 inference

Four fresh 2k runs screened 9.97M, 35.66M and 42.48M models and two rates.
The small .001 and larger .0003 models were continued to 6k. Their exact
Drive exports scored 16.04% (154/960) and 18.96% (182/960), with held-out
numeric agreement 38/245 and 41/245. The larger model's modest advantage cost
about 1.9x as much added training and export time. These short one-seed runs
do not establish an optimal size. Full evidence and receipts are in
experiments/short-pilots-v1 and experiments/finalist-pilots-v1.

Actual Tesla T4 comparisons selected inference batch 256. The established
180k model generated all 1,919 programs in 10.38 and 10.79 seconds; the larger
6k model took 20.05 and 20.56 seconds. All twelve measured runs matched their
already-scored JSONL files exactly. Median speedups over batch 64 were 2.73x
and 1.82x. See experiments/t4-comparison-v1 and its local-decode verification.
The best public score remains 53.85%; the latest bot ranking was 10/15.

To reduce Drive reconnects, the next CPU-prepared, bounded GPU session combines
a production-pipeline batch-speed probe, equal-example quality pilots only
if the batch changes, and a 20k-equivalent continuation stage. This is a plan,
not a completed result. All checkpoints must be saved to Drive and evaluated
in the bot; the session is capped at 45 minutes. No full-corpus run is started.

### Planned step-180k decoding check after measured T4 speedup

The actual T4 batch-256 baseline now takes about 10.6 seconds for all 1,919
questions. Revisit four-beam search on the unchanged established 180k model:
the older 120k held-out comparison gained 5/245 correct numeric answers, but
its T4 cost had been unknown. Compare greedy and four beams on the same
reserved training questions before any public submission. Keep weights,
precision, token limits, exact decoding, and syntax-only retry unchanged.
Record the primary beam count explicitly in prediction provenance. Prepare
and verify inputs on CPU; only run the full official test on T4 if the held-out
comparison improves. Require the full measured generation to remain below
180 seconds, then submit that one new exact JSONL and preserve its bot receipt.
The current grouped training job remains on its pinned source; this change
only prepares a later decoding experiment.

The matched production-path batch probe completed on Blackwell: two-round
median throughput was 1,794.57 examples/s at batch 64, 1,623.36 at 128,
1,433.72 at 256 and 1,319.89 at 512. Each pass consumed exactly 32,768
original examples and 2,419,826 target tokens. Batch 64 was 10.55% faster
than 128, with about 8.18 versus 15.78 GiB peak allocated memory.

Both short quality continuations added 262,144 example presentations from the
same own 6k parent. At batch 128, validation loss was 0.885387, held-out
numeric correctness 45/245 and public score 183/960 (19.06%). At batch 64,
loss was 0.877859, held-out correctness 41/245 and public score 182/960
(18.96%). The predefined held-out quality gate rejected batch 64 before its
bot score arrived. Continue at 128 from step 8048 to step 20000, keeping the
total budget at 2,560,000 example presentations on the selected lineage.
The faster batch was not adopted on speed or token loss alone. Both pilot
receipts were uploaded beside the exact predictions and read back from Drive.
The selected continuation remains in progress; no final gain is claimed yet.

At the established 180k checkpoint, the CPU held-out decoding comparison
matched 131/245 references with greedy generation and 137/245 with four beams;
execution failures fell from 19 to 11. Generation took 10.611 versus 50.372
seconds with four CPU threads. Model, evaluator, questions and output hashes
were checked. This small diagnostic gain supports a bounded actual-T4 test,
not an automatic change to the submission decoder. Evidence is retained in
experiments/decoding-180k and mirrored in the Drive benchmarks folder.

The next selected-lineage checkpoint, step 10000, regressed to 135/960 public
answers (14.06%) after the schedule restart. Its exact outputs and bot receipt
were saved and verified in Drive. A temporary effect of the restarted rate is
only a hypothesis; better token loss does not establish better code execution.
The current stage remains capped at step 20000 and is not evidence for extending
the budget. The established best remains 517/960 (53.85%), public rank 10/15
at the latest 16 October bot response. First place remains unachieved.
