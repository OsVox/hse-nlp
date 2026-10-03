"""Generate the course test answers with a trained encoder-decoder model."""

from __future__ import annotations

import ast
import csv
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .data import fetch_parquet


def generate_predictions(
    model_dir: Path,
    output_file: Path,
    cache_dir: Path,
    *,
    batch_size: int = 64,
    max_new_tokens: int = 192,
) -> tuple[int, float]:
    """Keep test order, save a CSV, and return (row count, generation seconds)."""
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Timing is not a T4 result: CUDA is unavailable")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    test_path = fetch_parquet("data/test/test.parquet", cache_dir)
    questions = pq.read_table(test_path, columns=["question"])["question"].to_pylist()
    answers = [""] * len(questions)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    # Include sorting, tokenization, generation, and CSV writing in the timing.
    lengths = [len(ids) for ids in tokenizer(questions, add_special_tokens=True)["input_ids"]]
    order = sorted(range(len(questions)), key=lambda index: lengths[index])
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            batch = tokenizer(
                [questions[index] for index in indices],
                padding=True, truncation=True, max_length=128,
                return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, answer in zip(indices, decoded):
                answers[index] = answer.strip()

        # Give malformed answers a second chance. The fallback only changes
        # programs that already fail Python parsing, so valid greedy outputs
        # keep their original decoding.
        invalid = []
        for index, answer in enumerate(answers):
            try:
                if not answer:
                    raise SyntaxError("empty program")
                ast.parse(answer)
            except SyntaxError:
                invalid.append(index)
        repaired = 0
        for offset in range(0, len(invalid), batch_size):
            indices = invalid[offset:offset + batch_size]
            batch = tokenizer(
                [questions[index] for index in indices],
                padding=True, truncation=True, max_length=128,
                return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            generated = model.generate(
                **batch, max_new_tokens=max_new_tokens,
                num_beams=4, num_return_sequences=4, do_sample=False,
                use_cache=True,
            )
            candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for position, index in enumerate(indices):
                for candidate in candidates[position * 4:(position + 1) * 4]:
                    candidate = candidate.strip()
                    if not candidate:
                        continue
                    try:
                        ast.parse(candidate)
                    except SyntaxError:
                        continue
                    answers[index] = candidate
                    repaired += 1
                    break
    if device.type == "cuda":
        torch.cuda.synchronize()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["code"])
        writer.writeheader()
        writer.writerows({"code": answer} for answer in answers)
    elapsed = time.monotonic() - started
    if not all(answers):
        raise RuntimeError("the model returned an empty answer for at least one question")
    print(
        f"generated={len(answers)} seconds={elapsed:.2f} device={device} "
        f"invalid_before_retry={len(invalid)} repaired={repaired}"
    )
    return len(answers), elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("homeworks/hw2/model"))
    parser.add_argument("--output-file", type=Path, default=Path("homeworks/hw2/predictions.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("homeworks/hw2/cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    generate_predictions(
        args.model_dir, args.output_file, args.cache_dir,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    )
