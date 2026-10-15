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
from .speed import group_batches, precision_dtype
from .checkpoint_io import archive_checkpoint, drive_preflight, package_jsonl, sha256

SOURCE_LENGTH = 128
TARGET_LENGTH = 192


def continuation_scheduler(optimizer, *, learning_rate, start_step, end_step, warmup):
    """Keep optimizer moments but start an explicit new learning-rate phase."""
    if end_step <= start_step:
        raise ValueError('Continuation must add training updates')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
        group['initial_lr'] = learning_rate

    def multiplier(phase_step):
        if phase_step < warmup:
            return (phase_step + 1) / max(1, warmup)
        progress = min(1.0, (phase_step - warmup) / max(1, end_step - start_step - warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


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
        output = model(**batch, use_cache=False)
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


def save_best_model(
    model: T5ForConditionalGeneration, tokenizer: PreTrainedTokenizerFast,
    step: int, token_loss: float, directory: Path,
) -> None:
    """Keep the best model on the held-out selection rows without optimizer state."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / "validation.json").write_text(
        json.dumps({"step": step, "token_loss": token_loss})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--group-window", type=int, default=32,
                        help="Число батчей для группировки по длине; 1 отключает её")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=SHARDS)
    parser.add_argument("--batches-per-shard", type=int, default=None)
    parser.add_argument("--mixed-rows-per-shard", type=int, default=None,
                        help="Mix a fresh random sample from all shards each epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=Path("homeworks/hw2/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("homeworks/hw2/checkpoints"))
    parser.add_argument("--drive-root", type=Path, default=None,
                        help="Mounted Drive run directory; verify before allocating the model")
    parser.add_argument("--predict-at-checkpoint", action="store_true",
                        help="Save model-generated test JSONL for every archived checkpoint")
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='Verified checkpoint from our own earlier run; use a new output directory')
    parser.add_argument('--restart-schedule', action='store_true',
                        help='Retain optimizer moments but warm up a new schedule over the remaining updates')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.max_shards) < 1:
        parser.error("steps, batch size, accumulation and max shards must be positive")
    if args.batches_per_shard is not None and args.batches_per_shard < 1:
        parser.error("batches per shard must be positive")
    if args.mixed_rows_per_shard is not None and args.mixed_rows_per_shard < 1:
        parser.error("mixed rows per shard must be positive")

    if args.group_window < 1:
        parser.error("group window must be positive")
    if args.predict_at_checkpoint and args.drive_root is None:
        parser.error('--predict-at-checkpoint requires --drive-root')
    if args.restart_schedule and args.resume_from is None:
        parser.error('--restart-schedule requires --resume-from')
    parent = None
    if args.resume_from is not None:
        if args.output_dir.exists():
            parser.error('--resume-from requires a new output directory')
        marker = args.resume_from / 'checkpoint.complete.json'
        parent = json.loads(marker.read_text())
        required = {'model.safetensors', 'training_state.pt', 'config.json', 'tokenizer.json', 'run_config.json', 'step.json'}
        if not required.issubset(parent['files']):
            raise ValueError('Parent checkpoint lacks resume files')
        for name, digest in parent['files'].items():
            if Path(name).name != name or sha256(args.resume_from / name) != digest:
                raise IOError(f'Parent checkpoint integrity failure: {name}')
        parent = {'path': str(args.resume_from), 'manifest': parent,
                  'run_config': json.loads((args.resume_from / 'run_config.json').read_text())}
    if args.drive_root is not None:
        drive_preflight(args.drive_root)

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = precision_dtype(args.precision, device)
    print(f"device={device} precision={dtype} group_window={args.group_window}", flush=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.resume_from or "homeworks/hw2/tokenizer")
    checkpoint = args.output_dir / "latest"
    load_checkpoint = args.resume_from or checkpoint
    previous_config_path = load_checkpoint / 'run_config.json'
    previous_config = json.loads(previous_config_path.read_text()) if previous_config_path.exists() else {}
    if previous_config.get('schedule_start') is not None and not args.restart_schedule:
        if args.steps != previous_config['steps']:
            parser.error('Changing a continuation budget requires --resume-from and --restart-schedule')
        args.learning_rate = previous_config['learning_rate']
        args.warmup = previous_config['warmup']
    if parent is None:
        parent = previous_config.get('parent_checkpoint')
    schedule_start = (json.loads((load_checkpoint / 'step.json').read_text())['step']
                      if args.restart_schedule else previous_config.get('schedule_start'))
    if schedule_start is not None and args.steps <= schedule_start:
        parser.error('Continuation must add updates beyond its schedule start')
    best_directory = args.output_dir / "best"
    best_record = best_directory / "validation.json"
    best_loss = json.loads(best_record.read_text())["token_loss"] if best_record.exists() else float("inf")
    if (load_checkpoint / "config.json").exists():
        model = T5ForConditionalGeneration.from_pretrained(load_checkpoint)
    else:
        model = new_model(tokenizer)
    model.to(device)
    model.train()
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run_config.json').write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'model_config': model.config.to_dict(),
        'heldout_exclusion': 'question text across all shards',
        'parent_checkpoint': parent,
        'schedule_start': schedule_start,
    }, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def multiplier(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    start_step = 0
    if (load_checkpoint / "training_state.pt").exists():
        state = torch.load(load_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        print(f"resumed_at_step={start_step}", flush=True)
    if args.restart_schedule:
        scheduler = continuation_scheduler(
            optimizer, learning_rate=args.learning_rate, start_step=start_step,
            end_step=args.steps, warmup=args.warmup,
        )
        print(f'new_schedule_start={start_step} peak_lr={args.learning_rate}', flush=True)
    elif schedule_start is not None:
        scheduler = continuation_scheduler(
            optimizer, learning_rate=previous_config['learning_rate'],
            start_step=schedule_start, end_step=previous_config['steps'],
            warmup=previous_config['warmup'],
        )
        scheduler.load_state_dict(state['scheduler'])
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = lr

    # A resumed run uses a new deterministic stream of the supplied train split.
    # Its optimizer state continues; it does not claim exact batch replay.
    batches = training_batches(
        args.cache_dir, args.batch_size, seed=args.seed + start_step,
        max_shards=args.max_shards, batches_per_shard=args.batches_per_shard,
        mixed_rows_per_shard=args.mixed_rows_per_shard,
    )
    batches = group_batches(batches, tokenizer, args.batch_size, args.group_window)
    validation = validation_examples(args.cache_dir)[:256]
    start_time = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        score = None
        optimizer.zero_grad(set_to_none=True)
        # Не копируем loss на CPU после каждого микробатча: это ждёт GPU.
        running_loss = torch.zeros((), device=device)
        for _ in range(args.accumulation):
            batch = encode_batch(next(batches), tokenizer, device)
            autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()
            with autocast:
                loss = model(**batch, use_cache=False).loss / args.accumulation
            scaler.scale(loss).backward()
            running_loss += loss.detach()
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
            if score < best_loss:
                best_loss = score
                save_best_model(model, tokenizer, step, score, best_directory)
                print(f"best_step={step} best_val_token_loss={score:.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, optimizer, scheduler, scaler, step, tokenizer, checkpoint)
            if args.drive_root is not None:
                if score is not None:
                    (checkpoint / 'validation.json').write_text(json.dumps({'step': step, 'token_loss': score}))
                else:
                    (checkpoint / 'validation.json').unlink(missing_ok=True)
                (checkpoint / 'run_config.json').write_text((args.output_dir / 'run_config.json').read_text())
                archived = args.drive_root / f'step-{step:06d}'
                archive_checkpoint(checkpoint, archived)
                print(f'drive_checkpoint_verified={archived}', flush=True)
                if args.predict_at_checkpoint:
                    from .inference import generate_predictions
                    output = archived / 'predictions.csv'
                    generate_predictions(archived, output, args.cache_dir,
                                         model=model, tokenizer=tokenizer, precision='fp32')
                    digest = package_jsonl(output, archived / 'predictions.jsonl')
                    (archived / 'results.complete.json').write_text(json.dumps({
                        'step': step, 'jsonl_sha256': digest,
                        'bot_status': 'pending', 'public_pass_at_1': None,
                        'question_order': 'official data/test/test.parquet',
                    }, indent=2))
                    model.train()
                    print(f'checkpoint_ready_for_bot={archived} jsonl_sha256={digest}', flush=True)
    print(f"finished_steps={args.steps}", flush=True)


if __name__ == "__main__":
    main()
