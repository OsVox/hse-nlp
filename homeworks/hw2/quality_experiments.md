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
