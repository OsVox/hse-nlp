"""Report held-out code quality without touching the course test labels."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .data import validation_examples


def evaluate(
    model_dir: Path, cache_dir: Path, rows: int = 256,
    batch_size: int = 32, offset: int = 256,
) -> dict:
    """Evaluate on held-out rows separate from checkpoint-selection rows 0:256."""
    if rows < 1 or batch_size < 1 or offset < 0:
        raise ValueError("rows and batch_size must be positive; offset must be nonnegative")
    examples = validation_examples(cache_dir)[offset:offset + rows]
    if len(examples) != rows:
        raise ValueError("requested evaluation rows exceed the held-out split")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = tokenizer(
                [row["question"] for row in examples[start:start + batch_size]],
                padding=True, truncation=True, max_length=128,
                return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            generated = model.generate(**batch, max_new_tokens=192, num_beams=1)
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    parsed = 0
    solve_defined = 0
    exact = 0
    for predicted, reference in zip(predictions, examples):
        exact += predicted.strip() == reference["code"].strip()
        if not predicted.strip():
            continue
        try:
            tree = ast.parse(predicted)
        except SyntaxError:
            continue
        parsed += 1
        solve_defined += any(
            isinstance(node, ast.FunctionDef) and node.name == "solve"
            for node in tree.body
        )
    result = {
        "validation_rows": len(examples),
        "validation_offset": offset,
        "syntax_valid_fraction": parsed / len(examples),
        "solve_defined_fraction": solve_defined / len(examples),
        "exact_match_fraction": exact / len(examples),
        "note": "These are diagnostics, not pass@1; no test answers are available.",
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("homeworks/hw2/model"))
    parser.add_argument("--cache-dir", type=Path, default=Path("homeworks/hw2/cache"))
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offset", type=int, default=256)
    args = parser.parse_args()
    evaluate(args.model_dir, args.cache_dir, args.rows, args.batch_size, args.offset)
