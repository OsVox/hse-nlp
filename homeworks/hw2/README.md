# Homework 2: question-to-code generation

This solution follows [the course brief](../hw2_code_generation.md). It trains a
randomly initialized encoder-decoder Transformer on the supplied
[`SlavaYus/tinysms_qa_12m` dataset](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m).
No pretrained weights, outside examples, or extra target programs are used.

## Files

- `data.py` downloads original Parquet shards, reserves 1,024 training rows for
  validation, and builds training batches.
- `tokenizer/` is an 8,192-token byte-level BPE tokenizer fitted only on the
  supplied training split, excluding those validation rows.
- `train.py` trains a 9.97-million-parameter T5-style model with four encoder
  and four decoder layers. It saves model and optimizer checkpoints.
- `evaluate.py` measures held-out syntax validity, presence of `solve`, and
  exact code match. These diagnostics are not pass@1.
- `inference.py` generates answers for the 1,919 test questions, preserves test
  order, writes one `code` column, and times the generation loop.
- `hw2_train_colab.ipynb` and `hw2_inference_colab.ipynb` run those scripts on a
  T4. The inference notebook checks the 180-second limit.
- `metrics.json` and `report.pdf` record results once the runs finish.

## Reproduce

Select an available GPU in Colab and run `hw2_train_colab.ipynb`. Download the final
`latest/` directory before the Colab runtime ends. Copy the model files into
`homeworks/hw2/model/` in this repository; `training_state.pt` is only needed
to resume training. Then run `hw2_inference_colab.ipynb` on a T4 to verify the
course's 180-second generation limit.

The training script can also be run from the repository root:

```bash
python -m homeworks.hw2.train --steps 20000 --batch-size 16 --accumulation 4
```

The baseline consumes each shuffled shard before moving to the next. For an
otherwise identical run that samples across more shards, add
`--batches-per-shard 1000` and use a separate output directory. This is an
experimental option; compare its held-out loss before choosing a final model.

The course brief says submission details for its Telegram bot will be
published later. The CSV packaging may need adjustment to those instructions.
No leaderboard score is claimed until the bot verifies a submission.
