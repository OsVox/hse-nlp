# Homework 2: question-to-code generation

This solution follows [the course brief](../hw2_code_generation.md). It trains a
randomly initialized encoder-decoder Transformer on the supplied
[`SlavaYus/tinysms_qa_12m` dataset](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m).
No pretrained weights, outside examples, or extra target programs are used.

Current verified public best: **583/960 (60.73%)**, step 75,776 of
`complete-curve-4k-v1`; latest bot rank is 10/16 at 23:15 (+03), 16 October.
The strict-greedy inference notebook pins this checkpoint and source. Its new
preparation bundle has not run; final T4 timing remains unverified.

## Full training-data coverage

The old `--mixed-rows-per-shard` sampler can repeat rows before visiting the
entire corpus. Availability of all 60 shards is not evidence of a complete pass.
The coverage audit bounds the 78M pilot at at most 2.21% unique-row coverage and
the previous best 73,024-step checkpoint at at most 78.97%; actual unique coverage is
unknown. Independent experiment models do not share training experience.

Prepare `full_coverage` indices on CPU, before allocating a GPU:

```bash
python -m homeworks.hw2.full_coverage --cache-dir DATA --output-dir INDEX
```

Use `train --coverage-index INDEX --input-pipeline encoded --max-shards 60`
with the desired training arguments, removing `--mixed-rows-per-shard` and other
sampling/filtering overrides. This optional path keeps original targets and
visits every eligible original row before repeating. It preserves the current
128/192 token limits; row coverage is not full-token exposure.

`coverage.json` records per-shard coverage and a consumed-batch cursor, tied to
the optimizer checkpoint step. Prefetched batches are excluded until an update
uses them. Resume preserves dataset, tokenizer, seed and grouping configuration.
FP32/BF16 are supported; FP16 is rejected until skipped-update retries are
implemented. Old models start a new tracked traversal with prior unique coverage
explicitly unknown. Exact data resume does not claim bitwise model-update replay.

The CPU preflight verified all 11,836,444 eligible row IDs exactly once, with no
held-out rows, duplicates or omissions. One pass requires 92,473 batches at 128.
Twenty-three focused tests passed, including tiny-model training and archived
checkpoint resume. Local CPU input preparation showed no obvious slowdown;
GPU throughput and score effects remain unmeasured. Evidence is in
`experiments/coverage-readiness-cpu-v1` and `experiments/full-coverage-cpu-v1`.
The longer 78M run remains on hold.

The subsequent `coverage-pipeline-1k-v1` GPU pilot measured the corrected loader,
but covered only a fraction of one pass. `corpus-token-cpu-v1` audited every
eligible row: the default limits cut 19,870 questions and 212,045 programs
(228,035 affected rows total). Raising the training target cap to 384 failed
the short comparison in `target-length-1k-v1`; this does not resolve the need
to preserve complete original programs.

The new opt-in `--untruncated-training` path retains complete questions and
programs, with the same logical row order and coverage accounting. It requires
`--coverage-index` and the encoded pipeline. Long logical batches are split
before GPU transfer using `--microbatch-token-budget` and
`--microbatch-attention-budget`; their gradients use the original logical-batch
loss denominator, including numeric-token weighting. Only a successful optimizer
update commits the logical batch's row receipt. An over-budget individual row
raises explicitly rather than being cut or skipped. Default capped experiments
and immutable prepared sources remain reproducible.

These budgets are memory proxies, not measured GPU byte limits. The bounded
GPU probe passed on G4, including 130 selected length extremes. Dropout and floating-point
rounding can differ between split and unsplit execution. Changing tokenization
mode rejects an old coverage cursor; start a clearly recorded new traversal.
CPU checks are in `experiments/untruncated-cpu-v1`. No CPU audit or small-model
test is evidence that the competition checkpoint completed a full training pass.

`complete-curve-4k-v1` then trained 3,776 updates with complete original sequences,
retaining four Drive-saved and bot-scored checkpoints. Public scores were
574, 581, 583 and 569/960. The selected best gained eight public answers, but
held-out gains were mixed (0/245 and +5/493, with three additional execution
failures). No automatic long extension follows. The best checkpoint tracked
352,256 unique original rows; the final checkpoint tracked 483,328, with no
repeats. Neither completed a full pass. The driver took 512.01 seconds, excluding
allocation, authorization and installation; median GPU utilization was 94%.
All checkpoint exports and bot receipts were verified; compute was released.

The remaining targeted screens are tracked in `experiments/scaling-release-v1`.
Adjacent means of two or three complete-sequence checkpoints improved the pilot
subset but failed confirmation; keep the original trained checkpoint. Increasing
complete-sequence chunk budgets to 2x/3x took 7.67%/8.69% more time in both-order
GPU probes, despite using more memory. Retain 1x budgets and effective batch 128.
The CPU results and the 26.70-second GPU measurement are saved to Drive; probe
weights were discarded and zero active Colab sessions verified. Next are size
replication and matched-compute confirmation, followed by the user-authorized
longer run with a verified complete pass. These tests do not establish an optimal
model size or exhaust all possible optimization methods.

## Files

- `data.py` downloads original Parquet shards, reserves 1,024 training rows for
  validation, excludes every copy of their question texts from future training,
  and builds sequential or globally mixed training batches.
- `tokenizer/` is an 8,192-token byte-level BPE tokenizer fitted only on the
  supplied training split, excluding those validation rows.
- `train.py` trains configurable T5-style models from scratch. The established
  model has 9.97M parameters; the larger selected model has 42.48M parameters
  with six encoder and four decoder layers. It saves resumable `latest/` checkpoints and the
  lowest-loss model on the 256 selection rows in `best/`.
- `evaluate.py` measures syntax validity, presence of `solve`, and exact code
  match on held-out rows 256–511, which are separate from the 256 rows used to
  select checkpoints. These diagnostics are not pass@1.
- `inference.py` defaults to strictly greedy decoding: one candidate, no sampling,
  no syntax retries. `--syntax-retry` exists only to reproduce historical runs.
  Exact output token IDs, question order, model hash, and timings are saved.
- `hw2_full_training_colab.ipynb` prepares the selected larger model on CPU
  and resumes it on G4. `hw2_inference_colab.ipynb` uses strict greedy decoding
  on the numeric-loss trial at 72k. `hw2_inference_vote_colab.ipynb` archives the earlier
  57.08% voting result. T4 timing remains deferred for the new configuration.
- `metrics.json` and `report.pdf` record results once the runs finish.

## Reproduce

Use the CPU-first Drive workflow below for training experiments. Keep GPU off
until file checks pass. Save checkpoints and exact predictions to Drive, export
from Drive, then submit the checked JSONL to the course bot.

`hw2_inference_colab.ipynb` loads the 42.48M numeric-loss step-072000 checkpoint from
Drive after CPU checks. Strict greedy decoding is the user's selected policy
as of 16 October. The historical 54.06% larger-model result, and the 53.85%
small-model result, used greedy decoding with syntax-only beam retries;
they must not be relabelled as strict greedy scores. Historical voting reached
57.08% (548/960), retained only as an experiment. Strict greedy exports need
separate bot receipts. The strict greedy 50k export scored 54.06% (519/960)
at 09:26 (+03:00), matching the earlier retry-enabled score. Save outputs
to Drive before export. Strict greedy at 60k scored 56.25% (540/960),
21 more correct than 50k. The final 70k greedy export scored **57.71% (554/960)** at 10:09 (+03:00), then the best result across our decoding experiments. The bot reported rank 8/15.

The completed training process used its original immutable source, whose automatic
checkpoint predictions allow syntax retries. Use `greedy_export.py` on saved
checkpoints before further submissions.

The selected 42.48M model was intentionally stopped at its durable 70,000-step
checkpoint after the user requested improvements beyond scaling. The GPU
session was released; the unchanged full-run extension was cancelled. Further training is limited to short matched tests of specific changes. The run had targeted
100,000 updates at batch 128 and BF16 on G4. Its CPU checks verified the original dataset,
model weights and Adam optimizer state before GPU allocation. It saves
weights, optimizer state, exact predictions and validation results every
10,000 steps. Each completed checkpoint must be exported from Drive,
verified and scored once. The saved original run plan is historical. See `experiments/full-42m-100k-v1/ready.json`
for the fixed plan and `started.json` for the observed launch, not final results.

The training script can also be run from the repository root:

```bash
python -m homeworks.hw2.train --steps 20000 --batch-size 16 --accumulation 4
```

The baseline consumes each shuffled shard before moving to the next. For an
otherwise identical run that samples across more shards, add
`--batches-per-shard 1000` and use a separate output directory. This is an
experimental option; compare its held-out loss before choosing a final model.

The bot accepts only JSONL with 1,919 exact model outputs. The inference notebook
converts the CSV without changing any program and verifies token provenance.
The earlier 18.85% result included a handwritten identifier repair that was
removed. The historical 57.08% voting result uses exact model outputs. The user confirmed
that official test.parquet row order is required. See quality_experiments.md
and experiments/ for controlled runs and the validation-copy caveat. First
place remains unmet; the PDF is a progress report.

## Оптимизации по материалам курса

См. [разбор приёмов, проверки и команды сравнения](optimizations.md).
Новые настройки обучения: `--group-window` и `--precision`; в инференсе
доступен `--precision` (по умолчанию `fp32`).

CPU-first Drive workflow: run `prepare_colab.prepare` before selecting a GPU.
It saves all 60 original training shards, official test questions, source code
and the historical baseline to Drive with hashes. Export the probe from Drive
and verify it locally. The GPU notebook restores those bundles even after a
fresh runtime; each 2,000-update checkpoint saves weights, optimizer state,
validation loss and exact model-generated JSONL in its own Drive folder.
Bot receipts are recorded beside each evaluated JSONL; pending is not a score.

Download finished artifacts directly with an authenticated local `gdrive` CLI;
do not allocate or reconnect a Colab runtime just to export files. Check
`gdrive account current`, locate the run with `gdrive files list --parent FOLDER_ID`,
then use `gdrive files download FILE_ID --destination LOCAL_DIRECTORY`.
Verify the downloaded archive's SHA-256 against the saved run record, and run
`verify_export.py` on each checkpoint before submitting. Authentication stays
in the CLI's local credential store, outside this repository.

## Improvements beyond scaling

`average_pilot.py` tests arithmetic means of our own 60k/70k and 50k/60k/70k
checkpoints on CPU. It compares strict greedy numeric agreement on held-out
rows 512:768 and confirms a selected winner on rows 768:1024. Both stages
require strictly more correct results and no extra execution failures. Only
a confirmed average gets a full test export; no new training data or targets
are created. The 60k/70k average improved the first sample from 142 to 152/247, but confirmation improved only 135 to 136/246 while execution failures rose from 10 to 13. It failed the predefined gate and was not submitted. The three-checkpoint average scored 141/247 and was rejected earlier. All models and results are retained in Drive and `experiments/checkpoint-average-70k-v1`. No GPU was used.

`loss_pilot.py` prepares a matched 2,000-step control and a factor-two numeric/operator token-weight experiment, both starting from the same 70k checkpoint and optimizer state. It preserves original targets, data stream, seed, model size, learning-rate schedule and strict greedy decoding. Five focused tests passed. CPU checks completed before GPU launch. Both 2,000-step trials finished in 453 seconds total (including restoration, evaluation and export), with median GPU activity of 97%. The unchanged-loss control scored 58.54% (562/960); numeric2 scored **59.69% (573/960)**, then our best public result, rank 8/15. This is an 11-answer gain at matched model size and training budget. Held-out results were mixed (138 vs 141/245, then 289 vs 284/493) with slightly more execution failures, so the predefined extension gate failed. Keep the best-scored artifact for reproduction, but do not extend this trial automatically. Colab showed no active sessions afterward. See `experiments/non-scaling-plan.json` for hypotheses and their status.

## Short inverse-frequency screen (16 October)

Three matched 1,024-update continuations from the best 72k parent completed and were bot-scored. Control: **575/960 (59.90%)**; inverse-square-root: 575/960; inverse: 558/960. Both sampling treatments lost held-out agreements and added execution failures, so neither is extended. The simpler control is pinned in the inference notebook. Current public rank: **9/16**, observed at 14:52 (+03:00). First place is not achieved.

All checkpoints and exact greedy outputs are on Drive, exported and verified. The successful driver took 367.16 seconds with median GPU activity 97%; allocation/OAuth overhead is separate. A prior failed attempt produced no checkpoint. The launcher now logs locally and requests runtime release even if Drive fails. No active Colab sessions remained. Evidence: `experiments/frequency-screen-1k-v1`. The updated inference notebook was statically checked; no new inference run or T4 timing is claimed.

## Comment cleanup continuation (16 October)

Removing comments is allowed and was tested again from the trained 72k checkpoint. Both arms received 1,024 updates. Control scored **575/960 (59.90%)**; comment/empty-line cleanup scored **560/960 (58.33%)**. Cleanup gained 3 held-out agreements in the first subset, tied the second and added 3 execution failures, failing the extension gate. No longer run starts. Existing best and inference pin stay unchanged; rank 9/16 was confirmed at 15:08 (+03:00).

The successful driver took 286.86 seconds, excluding allocation/authentication/setup, with median GPU activity 95%. All checkpoints and greedy outputs were saved to Drive, exported and verified, then bot-scored. CPU preflight was reused and 13 focused tests passed. Colab showed no active sessions. Evidence: `experiments/comment-continuation-1k-v1`.
