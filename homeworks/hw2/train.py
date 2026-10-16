"""Train a randomly initialized encoder-decoder Transformer on course data.

Example on a CUDA GPU:
  python -m homeworks.hw2.train --steps 30000 --batch-size 16 --accumulation 4

The reference programs are used exactly as supplied. No outside training
examples, pretrained weights, extra targets, or teacher outputs are used.
"""

from __future__ import annotations

import argparse
import hashlib
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
from .input_pipeline import encoded_batches, prefetch_batches, to_device
from .digit_tokenization import load_training_tokenizer
from .token_weighting import token_weights, weighted_loss
from .training_budget import TrainingClock, capture_rng, restore_rng

SOURCE_LENGTH = 128
TARGET_LENGTH = 192


def continuation_scheduler(optimizer, *, learning_rate, start_step, end_step, warmup):
    """Keep optimizer moments but start an explicit new learning-rate phase."""
    if end_step <= start_step:
        raise ValueError('Continuation must add training updates')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate * group.get('lr_scale', 1.)
        group['initial_lr'] = group['lr']

    def multiplier(phase_step):
        if phase_step < warmup:
            return (phase_step + 1) / max(1, warmup)
        progress = min(1.0, (phase_step - warmup) / max(1, end_step - start_step - warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def new_model(tokenizer: PreTrainedTokenizerFast, *, d_model=256, d_ff=768,
              encoder_layers=4, decoder_layers=4, num_heads=4,
              dropout=0.1, initialization_vocab_size=None,
              feed_forward_proj='gated-gelu') -> T5ForConditionalGeneration:
    if min(d_model, d_ff, encoder_layers, decoder_layers, num_heads) < 1:
        raise ValueError('Model dimensions must be positive')
    if d_model != 64 * num_heads or not 0 <= dropout < 1:
        raise ValueError('Use 64-dimensional attention heads and dropout in [0, 1)')
    if initialization_vocab_size is not None and initialization_vocab_size < len(tokenizer):
        raise ValueError('Initialization vocabulary must cover the actual vocabulary')
    if feed_forward_proj not in {'gated-gelu', 'gated-silu'}:
        raise ValueError('Unsupported feed-forward activation')
    config = T5Config(
        vocab_size=initialization_vocab_size or len(tokenizer),
        d_model=d_model,
        d_ff=d_ff,
        num_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        num_heads=num_heads,
        dropout_rate=dropout,
        feed_forward_proj=feed_forward_proj,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    model = T5ForConditionalGeneration(config)
    # Transformers 4.57 infers T5's cache depth from encoder num_layers.
    # A lazy full cache grows to the actual decoder depth and survives reload.
    if decoder_layers > encoder_layers:
        model.generation_config.cache_implementation = 'dynamic_full'
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return model


def tokenizer_for_run(checkpoint, requested=None):
    if requested is not None:
        if (checkpoint/'config.json').exists() or (checkpoint/'training_state.pt').exists():
            raise ValueError('A tokenizer override requires a fresh model')
        if not (requested/'tokenizer.json').is_file():
            raise ValueError('Tokenizer directory must contain tokenizer.json')
        return requested
    return checkpoint if (checkpoint/'tokenizer.json').exists() else Path('homeworks/hw2/tokenizer')


def parameter_hash(model, *, exclude_embeddings=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not exclude_embeddings or name != 'shared.weight':
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def nonembedding_hash(model):
    return parameter_hash(model, exclude_embeddings=True)


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
    *, coverage=None, decoder_generator=None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(),
        "decoder_generator": decoder_generator.get_state() if decoder_generator is not None else None,
    }, directory / "training_state.pt")
    (directory / "step.json").write_text(json.dumps({"step": step}))
    if coverage is not None:
        coverage_record = {**coverage.checkpoint(), 'optimizer_step': step}
        (directory / 'coverage.json').write_text(json.dumps(coverage_record, indent=2)+'\n')


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


def override_dropout(model, probability):
    """Change both module dropout and T5 attention's functional dropout."""
    if not 0 <= probability < 1:
        raise ValueError('dropout must be in [0, 1)')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = probability
        if isinstance(getattr(module, 'dropout', None), float):
            module.dropout = probability
        if hasattr(module, 'config'):
            module.config.dropout_rate = probability


def main() -> None:
    script_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=None)
    parser.add_argument('--d-ff', type=int, default=None)
    parser.add_argument('--encoder-layers', type=int, default=None)
    parser.add_argument('--decoder-layers', type=int, default=None)
    parser.add_argument('--num-heads', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--feed-forward-proj', choices=['gated-gelu','gated-silu'], default=None,
                        help='Fresh-model activation; persisted in the model configuration')
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--group-window", type=int, default=32,
                        help="Число батчей для группировки по длине; 1 отключает её")
    parser.add_argument('--input-pipeline', choices=['legacy', 'encoded'], default='legacy',
                        help='Reuse grouping token IDs instead of tokenizing each batch again')
    parser.add_argument('--prefetch-batches', type=int, default=0,
                        help='Bounded CPU preparation queue; requires the encoded pipeline')
    parser.add_argument('--shuffle-batch-order', action='store_true',
                        help='Shuffle completed length-grouped batches within each window')
    parser.add_argument('--individual-digits', action='store_true',
                        help='Split digits before BPE; preserve original vocabulary and target text')
    parser.add_argument('--tokenizer-dir', type=Path, default=None,
                        help='Own fitted tokenizer; fresh models only')
    parser.add_argument('--strip-comments', action='store_true',
                        help='Remove comments and empty lines from original train targets with an identical-AST check')
    parser.add_argument('--initialization-vocab-size', type=int, default=None,
                        help='Initialize at a common vocabulary size, then shrink; fresh matched trials only')
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument('--max-training-seconds', type=float, default=None,
                        help='Stop after a completed update; excludes checkpoint/evaluation work')
    parser.add_argument('--checkpoint-training-seconds', type=float, default=None,
                        help='Retain checkpoints at measured training-time intervals')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=SHARDS)
    parser.add_argument("--batches-per-shard", type=int, default=None)
    parser.add_argument("--mixed-rows-per-shard", type=int, default=None,
                        help="Mix a fresh random sample from all shards each epoch")
    parser.add_argument('--coverage-index', type=Path, default=None,
                        help='CPU-prepared full_coverage index; visit all eligible rows before repeating')
    parser.add_argument('--coverage-rows-per-shard', type=int, default=4096)
    parser.add_argument('--training-target-length', type=int, default=192,
                        help='Full-coverage training target cap; validation/generation remain 192, grouping uses 192')
    parser.add_argument('--untruncated-training', action='store_true',
                        help='Keep complete original questions and programs; requires full coverage')
    parser.add_argument('--microbatch-token-budget', type=int, default=40960,
                        help='Untruncated training: maximum padded source plus target tokens per GPU chunk')
    parser.add_argument('--microbatch-attention-budget', type=int, default=9961472,
                        help='Untruncated training: maximum batch*(source^2+target^2+source*target)')
    parser.add_argument('--frequency-index', type=Path, default=None)
    parser.add_argument('--frequency-power', type=float, default=0.0,
                        help='Accept original rows with probability count**(-power); CPU index required')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    parser.add_argument('--muon-learning-rate', type=float, default=None)
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Explicit new continuation: retain weights/step, reset optimizer, scaler and schedule')
    parser.add_argument('--numeric-token-weight', type=float, default=1.0,
                        help='Experimental loss weight for numeric/operator pieces; original targets stay unchanged')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--focal-gamma', type=float, default=0.0,
                        help='Downweight easily predicted original target tokens; zero preserves CE')
    parser.add_argument('--example-balanced-loss', action='store_true')
    parser.add_argument('--continuation-dropout', type=float, default=None)
    parser.add_argument('--decoder-input-dropout', type=float, default=0.0,
                        help='Training-only whole decoder embedding dropout; reference labels stay unchanged')
    parser.add_argument('--skip-truncated-targets', action='store_true')
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=Path("homeworks/hw2/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("homeworks/hw2/checkpoints"))
    parser.add_argument("--drive-root", type=Path, default=None,
                        help="Mounted Drive run directory; verify before allocating the model")
    parser.add_argument("--predict-at-checkpoint", action="store_true",
                        help="Save model-generated test JSONL for every archived checkpoint")
    parser.add_argument('--prediction-batch-size', type=int, default=64,
                        help='Generation batch for checkpoint exports; does not change training batches')
    parser.add_argument('--prediction-max-input-tokens', type=int, default=128)
    parser.add_argument('--prediction-max-new-tokens', type=int, default=192)
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='Verified checkpoint from our own earlier run; use a new output directory')
    parser.add_argument('--restart-schedule', action='store_true',
                        help='Retain optimizer moments but warm up a new schedule over the remaining updates')
    args = parser.parse_args()
    try:
        training_clock = TrainingClock(args.max_training_seconds, args.checkpoint_training_seconds)
    except ValueError as error:
        parser.error(str(error))
    if args.muon_learning_rate is not None and (args.optimizer != 'muon' or
            not math.isfinite(args.muon_learning_rate) or args.muon_learning_rate <= 0):
        parser.error('Positive Muon learning rate requires --optimizer muon')
    if not math.isfinite(args.numeric_token_weight) or args.numeric_token_weight < 1:
        parser.error('numeric token weight must be finite and at least one')
    if not 0 <= args.label_smoothing < 1:
        parser.error('label smoothing must be in [0, 1)')
    if not math.isfinite(args.focal_gamma) or args.focal_gamma < 0 or (args.focal_gamma and args.label_smoothing):
        parser.error('Nonnegative focal gamma requires zero label smoothing')
    if args.continuation_dropout is not None and (args.resume_from is None or not 0 <= args.continuation_dropout < 1):
        parser.error('continuation dropout requires a parent and a value in [0, 1)')
    if not math.isfinite(args.decoder_input_dropout) or not 0 <= args.decoder_input_dropout < 1:
        parser.error('decoder input dropout must be in [0, 1)')
    if args.skip_truncated_targets and args.input_pipeline != 'encoded':
        parser.error('target filtering requires encoded pipeline')
    if args.training_target_length < 192 or (args.training_target_length != 192 and args.coverage_index is None):
        parser.error('A longer training target limit requires the full-coverage pipeline; minimum is 192')
    if args.untruncated_training and (args.coverage_index is None or args.training_target_length != 192):
        parser.error('Untruncated training requires full coverage and no custom target cap')
    if min(args.microbatch_token_budget, args.microbatch_attention_budget) < 1:
        parser.error('Microbatch budgets must be positive')
    if args.shuffle_batch_order and args.input_pipeline != 'encoded':
        parser.error('batch-order shuffling requires encoded pipeline')
    if min(args.steps, args.batch_size, args.accumulation, args.max_shards) < 1:
        parser.error("steps, batch size, accumulation and max shards must be positive")
    if args.batches_per_shard is not None and args.batches_per_shard < 1:
        parser.error("batches per shard must be positive")
    if args.mixed_rows_per_shard is not None and args.mixed_rows_per_shard < 1:
        parser.error("mixed rows per shard must be positive")
    if args.coverage_index is not None and (
            args.input_pipeline != 'encoded' or args.coverage_rows_per_shard < 1 or
            args.mixed_rows_per_shard is not None or args.batches_per_shard is not None or
            args.frequency_index is not None or args.frequency_power or
            args.skip_truncated_targets or args.strip_comments):
        parser.error('Full coverage requires encoded input and unchanged, unfiltered original rows; remove legacy sampling options')
    if not 0 <= args.frequency_power <= 1 or (args.frequency_power and
            (args.frequency_index is None or args.mixed_rows_per_shard is None)):
        parser.error('frequency power in [0,1] requires index and mixed sampling when nonzero')
    frequency_index = None
    if args.frequency_index is not None:
        from .question_frequency import FrequencyIndex, digest
        frequency_index = FrequencyIndex(args.frequency_index, args.cache_dir, shards=args.max_shards)
        args.frequency_manifest_sha256 = digest(args.frequency_index/'manifest.json')

    if args.group_window < 1:
        parser.error("group window must be positive")
    if args.prefetch_batches < 0 or (args.prefetch_batches and args.input_pipeline != 'encoded'):
        parser.error('Non-negative prefetch depth requires --input-pipeline encoded')
    if args.predict_at_checkpoint and args.drive_root is None:
        parser.error('--predict-at-checkpoint requires --drive-root')
    if args.prediction_batch_size < 1:
        parser.error('--prediction-batch-size must be positive')
    if args.prediction_max_input_tokens < 0 or args.prediction_max_new_tokens < 1:
        parser.error('Prediction limits require nonnegative input and positive output lengths')
    if args.restart_schedule and args.resume_from is None:
        parser.error('--restart-schedule requires --resume-from')
    if args.reset_optimizer and not (args.resume_from is not None and args.restart_schedule):
        parser.error('--reset-optimizer requires --resume-from and --restart-schedule')
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
    if args.coverage_index is not None and dtype == torch.float16:
        parser.error('Full coverage requires fp32 or bf16; skipped fp16 updates need explicit batch retry support')
    print(f"device={device} precision={dtype} group_window={args.group_window}", flush=True)
    checkpoint = args.output_dir / "latest"
    load_checkpoint = args.resume_from or checkpoint
    tokenizer_path = tokenizer_for_run(load_checkpoint, args.tokenizer_dir)
    tokenizer = load_training_tokenizer(tokenizer_path, individual_digits=args.individual_digits)
    previous_config_path = load_checkpoint / 'run_config.json'
    previous_config = json.loads(previous_config_path.read_text()) if previous_config_path.exists() else {}
    if args.max_training_seconds is not None and previous_config and not args.restart_schedule:
        if any(getattr(args,key) != previous_config[key] for key in ('steps','warmup','learning_rate')):
            parser.error('Timed resume must preserve the original full-run learning-rate schedule')
    coverage = None
    if previous_config.get('coverage_index') and args.coverage_index is None:
        parser.error('Resume must preserve the full-coverage stream')
    if args.coverage_index is not None:
        from .full_coverage import FullCoverage, load_index
        manifest, paths, indices = load_index(args.cache_dir, args.coverage_index)
        if len(paths) != args.max_shards:
            parser.error('Coverage index shard count differs from --max-shards')
        coverage_path = load_checkpoint/'coverage.json'
        if previous_config.get('coverage_index') and not coverage_path.exists():
            raise ValueError('Full-coverage parent is missing its consumed-stream state')
        coverage_record = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
        if coverage_record is not None and coverage_record['optimizer_step'] != json.loads((load_checkpoint/'step.json').read_text())['step']:
            raise ValueError('Coverage cursor and optimizer checkpoint steps disagree')
        coverage_state = coverage_record['state'] if coverage_record is not None else None
        coverage_tokenizer = load_training_tokenizer(tokenizer_path, individual_digits=args.individual_digits)
        coverage = FullCoverage(manifest, paths, indices, coverage_tokenizer,
            seed=args.seed, batch_size=args.batch_size, window=args.group_window,
            rows_per_shard=args.coverage_rows_per_shard,
            target_length=args.training_target_length,
            untruncated=args.untruncated_training,
            shuffle_batch_order=args.shuffle_batch_order, state=coverage_state,
            start_step=json.loads((load_checkpoint/'step.json').read_text())['step']
                       if (load_checkpoint/'step.json').exists() else 0)
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
        if args.initialization_vocab_size is not None:
            parser.error('Common-vocabulary initialization requires a fresh model')
        if any(getattr(args, name) is not None for name in
               ['d_model', 'd_ff', 'encoder_layers', 'decoder_layers', 'num_heads', 'dropout', 'feed_forward_proj']):
            parser.error('Architecture overrides require a fresh model/output directory')
        model = T5ForConditionalGeneration.from_pretrained(load_checkpoint)
    else:
        model = new_model(tokenizer, initialization_vocab_size=args.initialization_vocab_size, **{
            name: getattr(args, name) for name in
            ['d_model', 'd_ff', 'encoder_layers', 'decoder_layers', 'num_heads', 'dropout', 'feed_forward_proj']
            if getattr(args, name) is not None})
    initial_nonembedding_sha256 = None
    initial_model_sha256 = None
    if args.initialization_vocab_size is not None:
        initial_nonembedding_sha256 = nonembedding_hash(model)
        initial_model_sha256 = parameter_hash(model)
        # Resizing consumes random draws. Restart the dropout stream identically.
        torch.manual_seed(args.seed)
    if args.continuation_dropout is not None:
        override_dropout(model, args.continuation_dropout)
    model.to(device)
    loss_weights = (token_weights(tokenizer, args.numeric_token_weight).to(device)
                    if args.numeric_token_weight != 1 or args.label_smoothing or args.example_balanced_loss or args.focal_gamma else None)
    split_loss_weights = loss_weights.detach().cpu() if args.untruncated_training and loss_weights is not None else None
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
        'initial_nonembedding_sha256': initial_nonembedding_sha256,
        'initial_model_sha256': initial_model_sha256,
    }, indent=2))

    from .optimizers import make_optimizer, restore_training_state
    if (load_checkpoint / 'training_state.pt').exists() and previous_config.get('optimizer', 'adamw') != args.optimizer and not args.reset_optimizer:
        parser.error('Changing optimizer requires --reset-optimizer; moments cannot be reused across optimizer types')
    optimizer = make_optimizer(model, args.optimizer, learning_rate=args.learning_rate,
                               muon_learning_rate=args.muon_learning_rate)

    def multiplier(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    start_step = 0
    resume_rng = None
    resume_decoder_rng = None
    if (load_checkpoint / "training_state.pt").exists():
        state = torch.load(load_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        start_step = restore_training_state(state, optimizer, scheduler, scaler,
                                            reset=args.reset_optimizer)
        resume_rng = state.get('rng')
        resume_decoder_rng = state.get('decoder_generator')
        if args.max_training_seconds is not None and not args.reset_optimizer and resume_rng is None:
            raise ValueError('Timed resume requires a checkpoint with saved RNG state')
        print(f"resumed_at_step={start_step} optimizer_reset={args.reset_optimizer}", flush=True)
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

    if (load_checkpoint / 'training_state.pt').exists():
        del state
    # Legacy sampling restarts at a new seed. Full coverage instead resumes its
    # saved consumed-batch cursor, independently of producer prefetch progress.
    batches = training_batches(
        args.cache_dir, args.batch_size, seed=args.seed + start_step,
        max_shards=args.max_shards, batches_per_shard=args.batches_per_shard,
        mixed_rows_per_shard=args.mixed_rows_per_shard,
        frequency_index=frequency_index, frequency_power=args.frequency_power,
    )
    if coverage is not None:
        batches = prefetch_batches(coverage.batches(), args.prefetch_batches,
                                  pin_memory=device.type == 'cuda')
    elif args.strip_comments:
        from .code_cleanup import cleaned_batches
        batches = cleaned_batches(batches)
    if coverage is None and args.input_pipeline == 'encoded':
        # Fast tokenizers temporarily change padding/truncation settings per call.
        # Keep the producer's instance separate from validation and prediction.
        input_tokenizer = (load_training_tokenizer(tokenizer_path,
            individual_digits=args.individual_digits) if args.prefetch_batches else tokenizer)
        batches = encoded_batches(batches, input_tokenizer, args.batch_size, args.group_window,
                                  skip_truncated_targets=args.skip_truncated_targets,
                                  shuffle_batch_seed=(args.seed + start_step if args.shuffle_batch_order else None))
        batches = prefetch_batches(batches, args.prefetch_batches, pin_memory=device.type == 'cuda')
    elif coverage is None:
        batches = group_batches(batches, tokenizer, args.batch_size, args.group_window)
    validation = validation_examples(args.cache_dir)[:256]
    decoder_generator = None
    if args.decoder_input_dropout:
        from .decoder_dropout import decoder_dropout_inputs
        decoder_generator = torch.Generator(device=device).manual_seed(args.seed + start_step + 2026101604)
        if resume_decoder_rng is not None:
            decoder_generator.set_state(resume_decoder_rng)
    if resume_rng is not None and not args.reset_optimizer:
        restore_rng(resume_rng)
    if start_step >= args.steps:
        raise ValueError('Training budget must exceed the saved step')
    start_time = time.monotonic()
    examples_processed = 0
    tokens_processed = torch.zeros(2, dtype=torch.int64, device=device)
    last_step = start_step
    saved_steps = []
    stop_reason = 'step_budget'
    for step in range(start_step + 1, args.steps + 1):
        training_clock.begin_update()
        score = None
        optimizer.zero_grad(set_to_none=True)
        # Не копируем loss на CPU после каждого микробатча: это ждёт GPU.
        running_loss = torch.zeros((), device=device)
        coverage_receipts = []
        for _ in range(args.accumulation):
            raw_batch = next(batches)
            if coverage is not None:
                coverage_receipts.append(raw_batch.pop('_coverage'))
            if args.untruncated_training:
                from .long_sequences import split_batch
                chunks = split_batch(raw_batch, token_budget=args.microbatch_token_budget,
                    attention_budget=args.microbatch_attention_budget, weights=split_loss_weights,
                    example_balanced=args.example_balanced_loss)
            else:
                chunks = [(raw_batch, 1.0)]
            for chunk, loss_fraction in chunks:
                batch = (to_device(chunk, device) if args.input_pipeline == 'encoded'
                         else encode_batch(chunk, tokenizer, device))
                examples_processed += batch['labels'].shape[0]
                tokens_processed[0] += batch['attention_mask'].sum()
                tokens_processed[1] += batch['labels'].ne(-100).sum()
                autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()
                with autocast:
                    decoder_options = (decoder_dropout_inputs(model, batch['labels'],
                        args.decoder_input_dropout, generator=decoder_generator)
                        if args.decoder_input_dropout else {})
                    output = model(**batch, **decoder_options, use_cache=False)
                    loss = (output.loss if loss_weights is None else
                            weighted_loss(output.logits, batch['labels'], loss_weights,
                                          smoothing=args.label_smoothing,
                                          example_balanced=args.example_balanced_loss,
                                          focal_gamma=args.focal_gamma)) * loss_fraction / args.accumulation
                scaler.scale(loss).backward()
                running_loss += loss.detach()
                del output, loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=coverage is not None)
        scaler.step(optimizer)
        scaler.update()
        for receipt in coverage_receipts:
            coverage.commit(receipt)
        scheduler.step()
        last_step = step
        if args.max_training_seconds is not None and device.type == 'cuda':
            torch.cuda.synchronize()
        timed_checkpoint, time_exhausted = training_clock.end_update()
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
        if step % args.save_every == 0 or step == args.steps or timed_checkpoint or time_exhausted:
            save_checkpoint(model, optimizer, scheduler, scaler, step, tokenizer, checkpoint,
                            coverage=coverage, decoder_generator=decoder_generator)
            saved_steps.append(step)
            (checkpoint / 'training-budget.json').write_text(json.dumps(dict(
                optimizer_step=step, training_seconds=training_clock.elapsed,
                completed_updates=step-start_step, examples_processed=examples_processed,
                max_training_seconds=args.max_training_seconds,
                planned_final_step=args.steps, time_limit_reached=time_exhausted,
                timing_scope='Completed updates including input waits; checkpoint/evaluation work excluded'),indent=2)+'\n')
            (checkpoint / 'run_config.json').write_text((args.output_dir / 'run_config.json').read_text())
            if coverage is not None:
                print('coverage='+json.dumps(coverage.checkpoint()['report']), flush=True)
            if args.drive_root is not None:
                if score is not None:
                    (checkpoint / 'validation.json').write_text(json.dumps({'step': step, 'token_loss': score}))
                else:
                    (checkpoint / 'validation.json').unlink(missing_ok=True)
                archived = args.drive_root / f'step-{step:06d}'
                archive_checkpoint(checkpoint, archived)
                print(f'drive_checkpoint_verified={archived}', flush=True)
                if args.predict_at_checkpoint:
                    from .inference import generate_predictions
                    output = archived / 'predictions.csv'
                    generate_predictions(archived, output, args.cache_dir,
                                         model=model, tokenizer=tokenizer, precision='fp32',
                                         batch_size=args.prediction_batch_size,
                                         max_input_tokens=args.prediction_max_input_tokens,
                                         max_new_tokens=args.prediction_max_new_tokens)
                    digest = package_jsonl(output, archived / 'predictions.jsonl')
                    (archived / 'results.complete.json').write_text(json.dumps({
                        'step': step, 'jsonl_sha256': digest,
                        'bot_status': 'pending', 'public_pass_at_1': None,
                        'question_order': 'official data/test/test.parquet',
                    }, indent=2))
                    model.train()
                    print(f'checkpoint_ready_for_bot={archived} jsonl_sha256={digest}', flush=True)
        if time_exhausted:
            stop_reason = 'training_time_budget'
            break
    if hasattr(batches, 'close'):
        batches.close()
    print(f"finished_steps={last_step}", flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    source_tokens, target_tokens = tokens_processed.tolist()
    budget = dict(examples_processed=examples_processed, source_tokens=source_tokens,
        target_tokens=target_tokens, updates=last_step-start_step,
        final_step=last_step, planned_final_step=args.steps, stop_reason=stop_reason,
        training_seconds=training_clock.elapsed, max_training_seconds=args.max_training_seconds,
        checkpoint_steps=saved_steps,
        loop_seconds_including_checkpoint_evaluation=time.monotonic()-start_time,
        process_seconds_including_setup=time.monotonic()-script_started,
        torch_version=torch.__version__, optimizer=args.optimizer)
    (args.output_dir/'compute-budget.json').write_text(json.dumps(budget, indent=2)+'\n')
    if args.drive_root is not None:
        (args.drive_root/'compute-budget.json').write_text(json.dumps(budget, indent=2)+'\n')


if __name__ == "__main__":
    main()
