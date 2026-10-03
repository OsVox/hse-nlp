"""Data access for the course's question-to-code dataset.

Only the supplied training split is used for fitting or validation. The
published test split is read solely by the inference notebook.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

DATASET = "SlavaYus/tinysms_qa_12m"
SHARDS = 60
VALIDATION_ROWS = 1_024
SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"]


def fetch_parquet(relative_path: str, cache_dir: Path) -> Path:
    """Cache one original Parquet file without modifying its records."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / Path(relative_path).name
    if destination.exists():
        return destination
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{relative_path}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    output.write(chunk)
        # Reject truncated files before they can be used in an experiment.
        pq.ParquetFile(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def training_shard(index: int, cache_dir: Path) -> Path:
    if not 0 <= index < SHARDS:
        raise ValueError(f"shard index must be between 0 and {SHARDS - 1}")
    return fetch_parquet(f"data/train/part-{index:05d}.parquet", cache_dir)


def validation_examples(cache_dir: Path) -> list[dict[str, str]]:
    table = pq.read_table(training_shard(0, cache_dir), columns=["question", "code"])
    return table.slice(0, VALIDATION_ROWS).to_pylist()


def train_tokenizer(cache_dir: Path, output_dir: Path, vocab_size: int = 8_192) -> Path:
    """Fit byte-level BPE on the supplied training split, excluding validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(training_shard(0, cache_dir), columns=["question", "code"])
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def texts() -> Iterator[str]:
        for row in table.slice(VALIDATION_ROWS).to_pylist():
            yield row["question"]
            yield row["code"]

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", eos_id)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
    )
    fast.save_pretrained(output_dir)
    return output_dir


def training_batches(
    cache_dir: Path, batch_size: int, *, seed: int = 42,
    max_shards: int = SHARDS, batches_per_shard: int | None = None,
) -> Iterator[list[dict[str, str]]]:
    """Yield shuffled batches, optionally rotating shards before exhaustion.

    A short run otherwise consumes just the first few shuffled shards. Limiting
    batches per shard spreads the same update budget across more of the supplied
    training split without adding data or reusing validation rows.
    """
    if batch_size < 1 or not 1 <= max_shards <= SHARDS:
        raise ValueError("invalid batch size or shard count")
    if batches_per_shard is not None and batches_per_shard < 1:
        raise ValueError("batches_per_shard must be positive")
    epoch = 0
    while True:
        shard_order = list(range(max_shards))
        random.Random(seed + epoch).shuffle(shard_order)
        for shard_index in shard_order:
            table = pq.read_table(
                training_shard(shard_index, cache_dir), columns=["question", "code"]
            )
            rows = table.slice(VALIDATION_ROWS if shard_index == 0 else 0).to_pylist()
            order = list(range(len(rows)))
            random.Random(seed + epoch * SHARDS + shard_index).shuffle(order)
            for batch_number, start in enumerate(range(0, len(order), batch_size)):
                if batches_per_shard is not None and batch_number >= batches_per_shard:
                    break
                yield [rows[index] for index in order[start:start + batch_size]]
        epoch += 1
