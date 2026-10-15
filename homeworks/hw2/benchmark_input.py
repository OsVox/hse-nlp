"""CPU equivalence gate and matched GPU probe for the input pipeline.

Probe weights are discarded. The only examples come from the previously
prepared official training pool, whose hash is recorded in each result.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .checkpoint_io import sha256
from .input_pipeline import encoded_batches, prefetch_batches, to_device
from .speed import group_batches
from .train import encode_batch

VARIANTS = ('legacy', 'encoded', 'encoded-prefetch')


def make_batches(rows, tokenizer, variant, batch_size=128, *, pin_memory=False):
    raw = (rows[i:i + batch_size] for i in range(0, len(rows), batch_size))
    if variant == 'legacy':
        return (encode_batch(b, tokenizer, torch.device('cpu'))
                for b in group_batches(raw, tokenizer, batch_size, 32))
    batches = encoded_batches(raw, tokenizer, batch_size, 32)
    return prefetch_batches(batches, 2 if variant == 'encoded-prefetch' else 0,
                            pin_memory=pin_memory)


def load_rows(pool, examples):
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    if not rows or examples < 1:
        raise ValueError('Need a nonempty official training pool and positive example count')
    return (rows * (examples // len(rows) + 1))[:examples]


def preflight(checkpoint, pool, output, examples):
    if torch.cuda.is_available():
        raise RuntimeError('Run equivalence checks on CPU before allocating GPU')
    rows = load_rows(pool, examples)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    result = {'pool_sha256': sha256(pool), 'examples': len(rows), 'batch_size': 128,
              'tokenizer_sha256': sha256(checkpoint/'tokenizer.json'), 'results': []}
    hashes = []
    for variant in VARIANTS:
        digest = hashlib.sha256()
        count = tokens = 0
        started = time.perf_counter()
        for batch in make_batches(rows, tokenizer, variant):
            count += batch['input_ids'].shape[0]
            tokens += int(batch['labels'].ne(-100).sum())
            for name, tensor in batch.items():
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
        assert count == len(rows)
        hashes.append(digest.hexdigest())
        result['results'].append({'variant': variant, 'tensor_sha256': hashes[-1],
                                  'examples': count, 'target_tokens': tokens,
                                  'cpu_preparation_seconds': time.perf_counter()-started})
    assert len(set(hashes)) == 1, 'Prepared batches differ from the legacy training path'
    result['status'] = 'all_tensors_equal'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def benchmark(checkpoint, pool, output, preflight_path, examples):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gate = json.loads(preflight_path.read_text())
    assert gate['status'] == 'all_tensors_equal'
    assert gate['pool_sha256'] == sha256(pool) and gate['examples'] == examples
    assert gate['tokenizer_sha256'] == sha256(checkpoint/'tokenizer.json')
    rows = load_rows(pool, examples)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    result = {'gpu': torch.cuda.get_device_name(0), 'pool_sha256': sha256(pool),
              'checkpoint_sha256': sha256(checkpoint/'model.safetensors'),
              'examples': examples, 'batch_size': 128, 'precision': 'bf16',
              'preflight': gate, 'results': []}
    output.mkdir(parents=True, exist_ok=False)
    (output/'probe.started.json').write_text(json.dumps(result, indent=2))
    for round_index, variants in enumerate((VARIANTS, VARIANTS[::-1])):
        for variant in variants:
            torch.manual_seed(42)
            torch.cuda.empty_cache()
            model = T5ForConditionalGeneration.from_pretrained(checkpoint).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)

            def transfer(batch):
                if variant == 'legacy':
                    return {k: v.to('cuda') for k, v in batch.items()}
                return to_device(batch, torch.device('cuda'))

            def update(batch):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch, use_cache=False).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                return loss

            warmup = make_batches(rows[:1280], tokenizer, variant, pin_memory=variant != 'legacy')
            for cpu_batch in warmup:
                loss = update(transfer(cpu_batch))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            telemetry = (output/f'gpu-round{round_index}-{variant}.csv').open('w')
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                        '--format=csv,noheader,nounits', '--loop-ms=200'], stdout=telemetry)
            batches = make_batches(rows, tokenizer, variant, pin_memory=variant != 'legacy')
            count = tokens = 0
            started = time.perf_counter()
            try:
                for cpu_batch in batches:
                    count += cpu_batch['input_ids'].shape[0]
                    tokens += int(cpu_batch['labels'].ne(-100).sum())
                    loss = update(transfer(cpu_batch))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                assert count == examples
                record = {'round': round_index, 'variant': variant, 'seconds': elapsed,
                          'examples': count, 'target_tokens': tokens,
                          'examples_per_second': count/elapsed, 'final_batch_loss': float(loss),
                          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
                result['results'].append(record)
                (output/'benchmark.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(record), flush=True)
            finally:
                batches.close()
                monitor.terminate(); monitor.wait(); telemetry.close()
                del model, optimizer, loss, cpu_batch
                gc.collect(); torch.cuda.empty_cache()
    speeds = {v: statistics.median(r['examples_per_second'] for r in result['results'] if r['variant'] == v)
              for v in VARIANTS}
    winner = max(speeds, key=speeds.get)
    # Require a meaningful measured gain before changing the production path.
    chosen = winner if speeds[winner] >= 1.05*speeds['legacy'] else 'legacy'
    selection = {'median_examples_per_second': speeds, 'chosen': chosen,
                 'rule': 'fastest median across reversed rounds; require at least 5% gain over legacy'}
    (output/'selection.json').write_text(json.dumps(selection, indent=2))
    print('INPUT_BENCHMARK_COMPLETE', json.dumps(selection), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-record', type=Path)
    parser.add_argument('--examples', type=int, default=32768)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.checkpoint, args.pool, args.output, args.examples)
    else:
        if args.preflight_record is None:
            parser.error('--preflight-record is required for a GPU probe')
        benchmark(args.checkpoint, args.pool, args.output, args.preflight_record, args.examples)
