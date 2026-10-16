"""Bounded, disposable throughput probe using the current Muon training path."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import torch
from transformers import PreTrainedTokenizerFast
from .benchmark_input import make_batches, load_rows
from .checkpoint_io import sha256
from .input_pipeline import to_device
from .optimizers import make_optimizer
from .train import new_model

SIZES = (128, 256, 512, 1024)
EXAMPLES = 8192

def prepare(pool, tokenizer, output):
    assert not torch.cuda.is_available()
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer, local_files_only=True)
    rows = load_rows(pool, EXAMPLES)
    counts = []
    for size in SIZES:
        stream = make_batches(rows, tok, 'encoded-prefetch', size)
        n = source = target = padded = 0
        try:
            for b in stream:
                n += len(b['labels'])
                source += int(b['attention_mask'].sum())
                target += int(b['labels'].ne(-100).sum())
                padded += b['input_ids'].numel() + b['labels'].numel()
        finally:
            stream.close()
        assert n == EXAMPLES
        counts.append(dict(batch_size=size, examples=n, source_tokens=source,
                           target_tokens=target, padded_tokens=padded))
    assert len({(r['source_tokens'], r['target_tokens']) for r in counts}) == 1
    output.write_text(json.dumps(dict(cpu_verified=True, pool_sha256=sha256(pool),
        tokenizer_sha256=sha256(tokenizer/'tokenizer.json'), counts=counts), indent=2))


def benchmark(pool, tokenizer, gate, output):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert hasattr(torch.optim, 'Muon')
    ready = json.loads(gate.read_text())
    assert ready['cpu_verified'] and ready['pool_sha256'] == sha256(pool)
    assert ready['tokenizer_sha256'] == sha256(tokenizer/'tokenizer.json')
    torch.set_num_threads(4)
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer, local_files_only=True)
    rows = load_rows(pool, EXAMPLES)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cpu_threads=4, precision='bf16', optimizer='Muon .001 / AdamW .0003',
        architecture='6 encoder / 4 decoder, width 512, FF 1536',
        pipeline='encoded-prefetch-2, group-window-32', cpu_gate=ready,
        probe_weights_discarded=True, accuracy_claim=False, runs=[])
    start = time.monotonic()
    telemetry = (output/'gpu.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits','--loop-ms=200'], stdout=telemetry)
    try:
        for repeat, sizes in enumerate((SIZES, SIZES[::-1])):
            for size in sizes:
                torch.manual_seed(42)
                model = new_model(tok,d_model=512,d_ff=1536,encoder_layers=6,
                    decoder_layers=4,num_heads=8,initialization_vocab_size=16384).cuda().train()
                opt = make_optimizer(model,'muon',learning_rate=.0003,muon_learning_rate=.001)
                stream = b = None
                record = dict(repeat=repeat,batch_size=size)
                def update(batch):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss = model(**to_device(batch,torch.device('cuda')),use_cache=False).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    opt.step()
                    return loss.detach()
                try:
                    stream = make_batches(rows[:size*3],tok,'encoded-prefetch',size,pin_memory=True)
                    for b in stream: update(b)
                    stream.close()
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    stream = make_batches(rows,tok,'encoded-prefetch',size,pin_memory=True)
                    n = tokens = padded = 0
                    for b in stream:
                        n += len(b['labels'])
                        tokens += int(b['attention_mask'].sum()+b['labels'].ne(-100).sum())
                        padded += b['input_ids'].numel()+b['labels'].numel()
                        last = update(b)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    assert n == EXAMPLES
                    expected = ready['counts'][0]
                    assert tokens == expected['source_tokens']+expected['target_tokens']
                    record.update(status='complete',seconds=elapsed,examples=n,nonpadding_tokens=tokens,
                        padding_fraction=1-tokens/padded,examples_per_second=n/elapsed,
                        useful_tokens_per_second=tokens/elapsed,final_loss=float(last),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                except torch.cuda.OutOfMemoryError:
                    record['status']='oom'
                finally:
                    if stream is not None:stream.close()
                    del opt,model,stream,b
                    gc.collect();torch.cuda.empty_cache()
                result['runs'].append(record)
                (output/'results.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(record),flush=True)
        medians = {size:statistics.median(r['examples_per_second'] for r in result['runs']
            if r['batch_size']==size) for size in SIZES
            if sum(r['batch_size']==size and r['status']=='complete' for r in result['runs'])==2}
        winner = max(medians,key=medians.get)
        result.update(completed=True,elapsed_seconds=time.monotonic()-start,
            median_examples_per_second=medians,
            throughput_candidate=winner if medians[winner]>=1.05*medians[128] else 128,
            caveat='Throughput only. Larger batches change update count and may affect learning; require a short quality comparison before adoption.')
        (output/'results.json').write_text(json.dumps(result,indent=2))
        print('COMPLETE',json.dumps(result),flush=True)
    finally:
        monitor.terminate();monitor.wait();telemetry.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    for name in ('pool','tokenizer','gate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.prepare:prepare(a.pool,a.tokenizer,a.gate)
    else:benchmark(a.pool,a.tokenizer,a.gate,a.output)
