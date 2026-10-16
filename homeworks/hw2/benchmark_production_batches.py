"""Compare training batches using the actual encoded/prefetched input path.

Probe updates are discarded. Equal examples are used at every batch size;
this measures throughput, not learning quality at a changed batch size.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .benchmark_input import load_rows, make_batches
from .checkpoint_io import sha256
from .input_pipeline import to_device

SIZES = (64, 128, 256, 512)


def prepare(checkpoint, pool, output, examples):
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    rows = load_rows(pool, examples)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    records = []
    for size in SIZES:
        count = tokens = 0
        stream = make_batches(rows, tokenizer, 'encoded-prefetch', size)
        try:
            for batch in stream:
                count += len(batch['input_ids'])
                tokens += int(batch['labels'].ne(-100).sum())
        finally:
            stream.close()
        assert count == examples
        records.append(dict(batch_size=size, examples=count, target_tokens=tokens))
    assert len({r['target_tokens'] for r in records}) == 1
    gate = dict(cpu_verified=True, pool_sha256=sha256(pool),
                model_sha256=sha256(checkpoint/'model.safetensors'),
                tokenizer_sha256=sha256(checkpoint/'tokenizer.json'),
                examples=examples, batches=records)
    output.write_text(json.dumps(gate, indent=2))
    print('BATCH_CPU_READY', json.dumps(gate), flush=True)


def benchmark(checkpoint, pool, output, gate_path):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gate = json.loads(gate_path.read_text())
    assert gate['cpu_verified'] and gate['pool_sha256'] == sha256(pool)
    assert gate['model_sha256'] == sha256(checkpoint/'model.safetensors')
    assert gate['tokenizer_sha256'] == sha256(checkpoint/'tokenizer.json')
    rows = load_rows(pool, gate['examples'])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(gpu=torch.cuda.get_device_name(), precision='bf16',
                  pipeline='encoded-prefetch', prefetch=2, group_batches=32,
                  gate=gate, probe_weights_discarded=True, runs=[])
    for repeat, sizes in enumerate((SIZES, SIZES[::-1])):
        for size in sizes:
            torch.manual_seed(42)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
            stream = batch = loss = None
            record = dict(repeat=repeat, batch_size=size)

            def update(cpu_batch):
                gpu_batch = to_device(cpu_batch, torch.device('cuda'))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**gpu_batch, use_cache=False).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                return loss

            try:
                stream = make_batches(rows[:size*10], tokenizer, 'encoded-prefetch', size, pin_memory=True)
                for batch in stream:
                    loss = update(batch)
                stream.close()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                stream = make_batches(rows, tokenizer, 'encoded-prefetch', size, pin_memory=True)
                count = tokens = 0
                for batch in stream:
                    count += len(batch['input_ids'])
                    tokens += int(batch['labels'].ne(-100).sum())
                    loss = update(batch)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                assert count == gate['examples'] and tokens == gate['batches'][0]['target_tokens']
                record.update(status='complete', seconds=elapsed, examples=count,
                              target_tokens=tokens, examples_per_second=count/elapsed,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              final_batch_loss=float(loss))
            except torch.cuda.OutOfMemoryError:
                record['status'] = 'oom'
            finally:
                if stream is not None:
                    stream.close()
                del optimizer, model, batch, loss, stream
                gc.collect()
                torch.cuda.empty_cache()
            result['runs'].append(record)
            (output/'results.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(record), flush=True)
    medians = {}
    for size in SIZES:
        records = [r for r in result['runs'] if r['batch_size']==size and r['status']=='complete']
        if len(records)==2:
            medians[size] = statistics.median(r['examples_per_second'] for r in records)
    winner = max(medians, key=medians.get)
    selected = winner if medians[winner] >= medians[128]*1.05 else 128
    result.update(completed=True, median_examples_per_second=medians, selected_batch=selected,
                  selection_rule='Two reversed rounds; require at least 5% gain over batch 128. Validate learning separately.')
    (output/'results.json').write_text(json.dumps(result, indent=2))
    print('BATCH_PROBE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--examples', type=int, default=32768)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.checkpoint, args.pool, args.output, args.examples)
    else:
        assert args.gate is not None
        benchmark(args.checkpoint, args.pool, args.output, args.gate)
