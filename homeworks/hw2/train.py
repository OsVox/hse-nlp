"""Train a randomly initialized encoder-decoder Transformer on course data.

Example on a CUDA GPU:
  python -m homeworks.hw2.train --steps 30000 --batch-size 16 --accumulation 4

The reference programs are used exactly as supplied. No outside training
examples, pretrained weights, extra targets, or teacher outputs are used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from .data import SHARDS, training_batches, validation_examples

SOURCE_LENGTH = 128
TARGET_LENGTH = 192


def new_model(tokenizer: PreTrainedTokenizerFast) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=256,
        d_ff=768,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=4,
        dropout_rate=0.1,
        feed_forward_proj="gated-gelu",
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return T5ForConditionalGeneration(config)


def encode_batch(
    examples: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sources = tokenizer(
        [row["question"] for row in examples],
        padding=True, truncation=True, max_length=SOURCE_LENGTH,
        return_tensors="pt",
    )
    targets = tokenizer(
        [row["code"] for row in examples],
        padding=True, truncation=True, max_length=TARGET_LENGTH,
        return_tensors="pt",
    )["input_ids"]
    # A truncated target still needs a proper stop token.
    target_lengths = targets.ne(tokenizer.pad_token_id).sum(dim=1)
    for row, length in enumerate(target_lengths.tolist()):
        targets[row, length - 1] = tokenizer.eos_token_id
    targets[targets == tokenizer.pad_token_id] = -100
    return {
        "input_ids": sources["input_ids"].to(device),
        "attention_mask": sources["attention_mask"].to(device),
        "labels": targets.to(device),
    }


@torch.no_grad()
def validation_loss(
    model: T5ForConditionalGeneration,
    examples: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    for start in range(0, len(examples), batch_size):
        batch = encode_batch(examples[start:start + batch_size], tokenizer, device)
        output = model(**batch)
        count = int(batch["labels"].ne(-100).sum())
        weighted_loss += float(output.loss) * count
        tokens += count
    model.train()
    return weighted_loss / tokens


def save_checkpoint(
    model: T5ForConditionalGeneration, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler,
    step: int, tokenizer: PreTrainedTokenizerFast, directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, directory / "training_state.pt")
    (directory / "step.json").write_text(json.dumps({"step": step}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=SHARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=Path("homeworks/hw2/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("homeworks/hw2/checkpoints"))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.max_shards) < 1:
        parser.error("steps, batch size, accumulation and max shards must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device={device}", flush=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("homeworks/hw2/tokenizer")
    checkpoint = args.output_dir / "latest"
    if (checkpoint / "config.json").exists():
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = new_model(tokenizer)
    model.to(device)
    model.train()
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def multiplier(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    if (checkpoint / "training_state.pt").exists():
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        print(f"resumed_at_step={start_step}", flush=True)

    # A resumed run uses a new deterministic stream of the supplied train split.
    # Its optimizer state continues; it does not claim exact batch replay.
    batches = training_batches(
        args.cache_dir, args.batch_size, seed=args.seed + start_step,
        max_shards=args.max_shards,
    )
    validation = validation_examples(args.cache_dir)[:256]
    start_time = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(args.accumulation):
            batch = encode_batch(next(batches), tokenizer, device)
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with autocast:
                loss = model(**batch).loss / args.accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            elapsed = time.monotonic() - start_time
            print(f"step={step} train_loss={running_loss:.4f} elapsed_s={elapsed:.1f}", flush=True)
        if step % args.eval_every == 0:
            score = validation_loss(model, validation, tokenizer, device, args.batch_size)
            print(f"step={step} val_token_loss={score:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, optimizer, scheduler, scaler, step, tokenizer, checkpoint)
    print(f"finished_steps={args.steps}", flush=True)


if __name__ == "__main__":
    main()
