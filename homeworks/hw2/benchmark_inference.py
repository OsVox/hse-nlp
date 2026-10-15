"""Bounded, paired inference batch-size comparison using one immutable model.

Prepare files on CPU first. Run this on an otherwise idle GPU, then copy every
result to Drive before exporting. This script never trains or submits to a bot.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
import transformers
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .checkpoint_io import package_jsonl, sha256
from .inference import generate_predictions
from .speed import precision_dtype


def programs(path):
    with path.open(newline='', encoding='utf-8') as f:
        return [row['code'] for row in csv.DictReader(f)]


def benchmark(model_dir, cache_dir, output_dir, expected_hash, batches, precision,
              budget_seconds=600, profile=False):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU timing is not evidence for T4')
    if sha256(model_dir/'model.safetensors') != expected_hash:
        raise ValueError('Checkpoint hash mismatch')
    if not (cache_dir/'test.parquet').exists():
        raise ValueError('Prepare and verify the test file before allocating GPU')
    if not batches or batches[0] != 64 or any(b < 1 for b in batches):
        raise ValueError('First batch size must be the baseline 64')
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    dtype = precision_dtype(precision, device)
    t0 = time.monotonic()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True).to(device).eval()
    torch.cuda.synchronize()
    record = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  transformers=transformers.__version__, precision=precision,
                  model_sha256=expected_hash, test_sha256=sha256(cache_dir/'test.parquet'),
                  script_sha256=sha256(Path(__file__)),
                  inference_sha256=sha256(Path(__file__).with_name('inference.py')),
                  model_load_seconds=time.monotonic()-t0, runs=[],
                  timing_scope='Generation timer includes tokenization, sorting, greedy, '
                  'syntax retry, decode and local CSV. Call timer additionally includes '
                  'test reading, provenance hashing/writing and output verification. '
                  'Model load and Drive archival are reported separately/excluded.',
                  max_new_tokens=192, max_input_tokens=128,
                  budget_seconds=budget_seconds, automatic_submission=False)

    def persist():
        temporary = output_dir/'results.partial.json'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(output_dir/'results.json')

    # An explicit small warm-up, retained separately from measured full passes.
    start = time.monotonic()
    batch = tokenizer(['Add two numbers.']*8, return_tensors='pt',
                      return_token_type_ids=False).to(device)
    with torch.inference_mode():
        model.generate(**batch, max_new_tokens=16, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    record['warmup_seconds'] = time.monotonic()-start
    persist()
    reference = None
    skipped = set()
    run_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='hw2-inference-') as temporary:
        local = Path(temporary)
        for repeat, order in enumerate([batches, list(reversed(batches))]):
            for size in order:
                if size in skipped:
                    continue
                if time.monotonic()-run_start >= budget_seconds:
                    record['stopped_at_budget'] = True
                    persist()
                    return record
                name = f'{precision}-b{size}-r{repeat}'
                destination = local/f'{name}.csv'
                torch.cuda.reset_peak_memory_stats()
                began = time.monotonic()
                try:
                    count, seconds = generate_predictions(
                        model_dir, destination, cache_dir, batch_size=size,
                        precision=precision, model=model, tokenizer=tokenizer)
                    torch.cuda.synchronize()
                    call_seconds = time.monotonic()-began
                except torch.cuda.OutOfMemoryError:
                    record['runs'].append(dict(batch_size=size, repeat=repeat, status='oom'))
                    skipped.add(size)
                    torch.cuda.empty_cache()
                    persist()
                    continue
                actual = programs(destination)
                if reference is None:
                    reference = actual
                assert count == len(reference) == 1919
                digest = package_jsonl(destination, destination.with_suffix('.jsonl'))
                entry = dict(batch_size=size, repeat=repeat, status='complete',
                             generation_seconds=seconds, call_seconds=call_seconds,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             changed_from_baseline=sum(a != b for a,b in zip(actual,reference)),
                             jsonl_sha256=digest, rows=count, files={})
                for path in sorted(local.glob(name+'.*')):
                    target = output_dir/path.name
                    shutil.copyfile(path,target)
                    entry['files'][path.name] = sha256(target)
                    assert entry['files'][path.name] == sha256(path)
                record['runs'].append(entry)
                persist()
                print(json.dumps(entry),flush=True)
        record['summary'] = []
        for size in batches:
            rows = [r for r in record['runs'] if r['batch_size']==size and r['status']=='complete']
            if len(rows)==2:
                record['summary'].append(dict(
                    batch_size=size,
                    median_seconds=statistics.median(r['generation_seconds'] for r in rows),
                    max_seconds=max(r['generation_seconds'] for r in rows),
                    all_outputs_equal_baseline=all(r['changed_from_baseline']==0 for r in rows),
                    repeat_outputs_identical=rows[0]['jsonl_sha256']==rows[1]['jsonl_sha256']))
        if profile and time.monotonic()-run_start < budget_seconds:
            # Diagnostic pass has synchronisation overhead; never include it in medians.
            original = model.generate
            calls = []
            def measured_generate(*args, **kwargs):
                torch.cuda.synchronize(); begin = time.monotonic()
                result = original(*args, **kwargs)
                torch.cuda.synchronize()
                calls.append(dict(beams=kwargs.get('num_beams',1),
                                  batch_size=kwargs['input_ids'].shape[0],
                                  generated_width=result.shape[1],
                                  seconds=time.monotonic()-begin))
                return result
            model.generate = measured_generate
            try:
                path = local/'profile.csv'
                _, seconds = generate_predictions(model_dir,path,cache_dir,batch_size=64,
                    precision=precision,model=model,tokenizer=tokenizer)
                record['profile'] = dict(calls=calls, total_seconds=seconds,
                    outputs_equal_baseline=programs(path)==reference,
                    note='Diagnostic synchronization overhead; excluded from timing comparison.')
            finally:
                model.generate = original
        record['completed'] = True
        persist()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-model-sha256',required=True)
    parser.add_argument('--batches',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    parser.add_argument('--budget-seconds',type=float,default=600)
    parser.add_argument('--profile',action='store_true')
    args = parser.parse_args()
    benchmark(args.model_dir,args.cache_dir,args.output_dir,args.expected_model_sha256,
              args.batches,args.precision,args.budget_seconds,args.profile)
