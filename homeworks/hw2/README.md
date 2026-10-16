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
- `train.py` trains a 9.97-million-parameter T5-style model with four encoder
  and four decoder layers. It saves resumable `latest/` checkpoints and the
  lowest-loss model on the 256 selection rows in `best/`.
- `evaluate.py` measures syntax validity, presence of `solve`, and exact code
  match on held-out rows 256–511, which are separate from the 256 rows used to
  select checkpoints. These diagnostics are not pass@1.
- `inference.py` generates answers for the 1,919 test questions, retries only
  syntactically invalid programs with four beams, preserves each selected model
  output exactly, records its token IDs and checkpoint hash, preserves test order, writes one `code` column,
  and times the full generation loop.
- `hw2_train_colab.ipynb` and `hw2_inference_colab.ipynb` run those scripts on a
  T4. The inference notebook checks the 180-second limit.
- `metrics.json` and `report.pdf` record results once the runs finish.

## Reproduce

Use the CPU-first Drive workflow below for training experiments. Keep GPU off
until file checks pass. Save checkpoints and exact predictions to Drive, export
from Drive, then submit the checked JSONL to the course bot.

`hw2_inference_colab.ipynb` loads the established step-180000 checkpoint from
Drive. Run its CPU preparation phase first, then its T4 phase. Batch 256 was
measured at 10.38 and 10.79 seconds on an actual Tesla T4 on 16 October; all
1,919 programs exactly match the JSONL scored at 53.85% (517/960). It is about
2.73 times faster than batch 64 by the two-run medians. The GPU phase saves
results to Drive and requests release; verify that Colab actually disconnects.
Future weights need a new timing and output check.

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
removed. The current 53.85% result uses exact model outputs. The user confirmed
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
