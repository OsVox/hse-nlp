"""CPU-prepared, bounded training throughput probe using only allowed examples."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time


def prepare(cache: Path, output: Path):
    import torch
    from .data import training_batches
    if torch.cuda.is_available():
        raise RuntimeError('Prepare the probe examples on CPU before allocating GPU')
    rows = next(training_batches(cache, 60 * 256, mixed_rows_per_shard=256, seed=710))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(f'PREPARED {len(rows)} original training examples: {output}', flush=True)


def benchmark(checkpoint: Path, pool: Path, output: Path, measured_examples: int):
    import torch
    from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
    from .train import encode_batch
    from .speed import group_batches
    from .checkpoint_io import sha256

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    rows = (rows * (measured_examples // len(rows) + 1))[:measured_examples]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    result = {'gpu': torch.cuda.get_device_name(0), 'parent_model_sha256': sha256(checkpoint/'model.safetensors'),
              'pool_sha256': sha256(pool), 'precision': 'bf16', 'warmup_updates': 10,
              'measured_examples': measured_examples, 'results': []}
    (output/'probe.started.json').write_text(json.dumps(result, indent=2))
    for round_index, sizes in enumerate(([64,128,256,512], [512,256,128,64])):
        for batch_size in sizes:
            torch.manual_seed(42)
            torch.cuda.empty_cache()
            batch = loss = None
            model = T5ForConditionalGeneration.from_pretrained(checkpoint).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
            def stream():
                while True:
                    for start in range(0, len(rows), batch_size):
                        yield rows[start:start+batch_size]
            batches = group_batches(stream(), tokenizer, batch_size, 32)
            record = {'round': round_index, 'batch_size': batch_size}
            telemetry = (output/f'gpu-round{round_index}-batch{batch_size}.csv').open('w')
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits','--loop-ms=200'],stdout=telemetry,stderr=subprocess.DEVNULL)
            try:
                for _ in range(10):
                    batch=encode_batch(next(batches), tokenizer, torch.device('cuda'))
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss=model(**batch, use_cache=False).loss
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                # Reset the stream: every size measures the same examples exactly once.
                raw = (rows[start:start+batch_size] for start in range(0, len(rows), batch_size))
                batches = group_batches(raw, tokenizer, batch_size, 32)
                examples=tokens=0; preparation=compute=0.0; started=time.perf_counter()
                while examples < len(rows):
                    t=time.perf_counter()
                    batch=encode_batch(next(batches),tokenizer,torch.device('cuda'))
                    examples+=batch['input_ids'].shape[0]
                    tokens+=int(batch['labels'].ne(-100).sum())
                    preparation+=time.perf_counter()-t
                    begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    begin.record(); optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=model(**batch,use_cache=False).loss
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                    end.record(); end.synchronize(); compute+=begin.elapsed_time(end)/1000
                elapsed=time.perf_counter()-started
                record.update(status='ok',seconds=elapsed,examples=examples,target_tokens=tokens,
                              examples_per_second=examples/elapsed,target_tokens_per_second=tokens/elapsed,
                              preparation_seconds=preparation,gpu_compute_seconds=compute,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              final_batch_loss=float(loss))
            except torch.cuda.OutOfMemoryError:
                record['status']='out_of_memory'
            finally:
                monitor.terminate(); monitor.wait(); telemetry.close()
                del model,optimizer,batch,loss
                gc.collect(); torch.cuda.empty_cache()
            result['results'].append(record)
            temporary=output/'benchmark.json.partial'; temporary.write_text(json.dumps(result,indent=2)); temporary.replace(output/'benchmark.json')
            print(json.dumps(record),flush=True)
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path('/content/hw2_cache'))
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--examples',type=int,default=16384)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.cache_dir,args.pool)
    else:
        benchmark(args.checkpoint,args.pool,args.output,args.examples)
