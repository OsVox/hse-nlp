# Homework 2: question-to-code generation

This solution follows [the course brief](../hw2_code_generation.md). It trains a
randomly initialized encoder-decoder Transformer on the supplied
[`SlavaYus/tinysms_qa_12m` dataset](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m).
No pretrained weights, outside examples, or extra target programs are used.

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
  on its 70k checkpoint. `hw2_inference_vote_colab.ipynb` archives the earlier
  57.08% voting result. T4 timing remains deferred for the new configuration.
- `metrics.json` and `report.pdf` record results once the runs finish.

## Reproduce

Use the CPU-first Drive workflow below for training experiments. Keep GPU off
until file checks pass. Save checkpoints and exact predictions to Drive, export
from Drive, then submit the checked JSONL to the course bot.

`hw2_inference_colab.ipynb` loads the 42.48M step-070000 checkpoint from
Drive after CPU checks. Strict greedy decoding is the user's selected policy
as of 16 October. The historical 54.06% larger-model result, and the 53.85%
small-model result, used greedy decoding with syntax-only beam retries;
they must not be relabelled as strict greedy scores. Historical voting reached
57.08% (548/960), retained only as an experiment. Strict greedy exports need
separate bot receipts. The strict greedy 50k export scored 54.06% (519/960)
at 09:26 (+03:00), matching the earlier retry-enabled score. Save outputs
to Drive before export. Strict greedy at 60k scored 56.25% (540/960),
21 more correct than 50k. The final 70k greedy export scored **57.71% (554/960)** at 10:09 (+03:00), now the best result across all our decoding experiments. The bot reported rank 8/15.

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

`loss_pilot.py` prepares a matched 2,000-step control and a factor-two numeric/operator token-weight experiment, both starting from the same 70k checkpoint and optimizer state. It preserves original targets, data stream, seed, model size, learning-rate schedule and strict greedy decoding. Five focused tests passed. CPU file and optimizer checks must finish before GPU launch. No longer run starts automatically. See `experiments/non-scaling-plan.json` for hypotheses and their status.
