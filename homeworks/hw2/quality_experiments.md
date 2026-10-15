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
