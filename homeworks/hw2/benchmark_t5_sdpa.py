"""CPU numerical gate and bounded disposable GPU attention comparison."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from transformers import PreTrainedTokenizerFast

from .benchmark_input import make_batches
from .checkpoint_io import sha256
from .input_pipeline import to_device
from .optimizers import make_optimizer
from .t5_sdpa import enable_sdpa_probe
from .train import new_model, encode_batch

EXAMPLES=8192
BATCH=128


def make_model(tok,dropout=.1):
    return new_model(tok,d_model=512,d_ff=1536,encoder_layers=6,
                     decoder_layers=4,num_heads=8,dropout=dropout)


def cpu_gate(pool,tokenizer,output):
    assert not torch.cuda.is_available()
    torch.set_num_threads(2)
    rows=[json.loads(line) for line in pool.read_text().splitlines()]
    assert len(rows)==EXAMPLES
    tok=PreTrainedTokenizerFast.from_pretrained(tokenizer,local_files_only=True)
    torch.manual_seed(42);native=make_model(tok,dropout=0);candidate=deepcopy(native)
    count=enable_sdpa_probe(candidate)
    batch=encode_batch(rows[:4],tok,torch.device('cpu'))
    a=native(**batch,use_cache=False);b=candidate(**batch,use_cache=False)
    torch.testing.assert_close(a.logits,b.logits,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(a.loss,b.loss,atol=2e-6,rtol=1e-6)
    a.loss.backward();b.loss.backward();max_grad=0.;bias_grad={}
    expected=dict(native.named_parameters())
    for name,p in candidate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,expected[name].grad,atol=3e-6,rtol=1e-3,msg=name)
        max_grad=max(max_grad,float((p.grad-expected[name].grad).abs().max()))
        if 'relative_attention_bias' in name:
            bias_grad[name]=float(p.grad.abs().sum());assert bias_grad[name]>0
    result=dict(status='cpu_numerical_gate_passed',parameters=sum(p.numel() for p in native.parameters()),
        patched_attention_modules=count,pool_sha256=sha256(pool),examples=EXAMPLES,
        tokenizer_sha256=sha256(tokenizer/'tokenizer.json'),
        adapter_sha256=sha256(Path(__file__).with_name('t5_sdpa.py')),
        native_loss=float(a.loss.detach()),sdpa_loss=float(b.loss.detach()),
        max_logit_abs_difference=float((a.logits-b.logits).abs().max().detach()),
        max_parameter_gradient_abs_difference=max_grad,relative_bias_grad_l1=bias_grad,
        dropout_disabled_for_numeric_comparison=True,synthetic_training_data_used=False)
    del a,b;native.zero_grad(set_to_none=True);candidate.zero_grad(set_to_none=True)
    native.eval();candidate.eval();inputs={k:v for k,v in batch.items() if k!='labels'}
    with torch.no_grad():
        reference=native.generate(**inputs,max_new_tokens=16,num_beams=1,do_sample=False)
        for cache in [False,True]:
            actual=candidate.generate(**inputs,max_new_tokens=16,num_beams=1,do_sample=False,use_cache=cache)
            assert torch.equal(reference,actual)
    result['greedy_cached_uncached_equal']=True
    stream=make_batches(rows,tok,'encoded-prefetch',BATCH)
    examples=source=target=padded=0
    try:
        for batch in stream:
            examples+=len(batch['labels']);source+=int(batch['attention_mask'].sum())
            target+=int(batch['labels'].ne(-100).sum())
            padded+=batch['input_ids'].numel()+batch['labels'].numel()
    finally:stream.close()
    assert examples==EXAMPLES
    result['counts']=dict(examples=examples,source_tokens=source,target_tokens=target,padded_tokens=padded)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def benchmark(pool,tokenizer,gate,output):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    ready=json.loads(gate.read_text())
    assert ready['status']=='cpu_numerical_gate_passed'
    assert ready['pool_sha256']==sha256(pool) and ready['tokenizer_sha256']==sha256(tokenizer/'tokenizer.json')
    assert ready['adapter_sha256']==sha256(Path(__file__).with_name('t5_sdpa.py'))
    tok=PreTrainedTokenizerFast.from_pretrained(tokenizer,local_files_only=True)
    rows=[json.loads(line) for line in pool.read_text().splitlines()];assert len(rows)==EXAMPLES
    output.mkdir(parents=True,exist_ok=False)
    result=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cpu_gate=ready,
        batch_size=BATCH,optimizer='Muon .001 / AdamW .0003',precision='bf16',
        source='Official eligible original shard-0 rows; pool hash and row IDs preserved',
        weights_discarded=True,quality_claim=False,production_changed=False,runs=[])
    began=time.monotonic();telemetry=(output/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=telemetry)
    try:
        for repeat,variants in enumerate([('native','sdpa'),('sdpa','native')]):
            for variant in variants:
                torch.manual_seed(42);model=make_model(tok).cuda().train()
                if variant=='sdpa':enable_sdpa_probe(model)
                optimizer=make_optimizer(model,'muon',learning_rate=.0003,muon_learning_rate=.001)
                stream=batch=last=None;record=dict(repeat=repeat,variant=variant)
                def update(batch):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=model(**to_device(batch,torch.device('cuda')),use_cache=False).loss
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();return loss.detach()
                try:
                    stream=make_batches(rows[:3*BATCH],tok,'encoded-prefetch',BATCH,pin_memory=True)
                    for batch in stream:last=update(batch)
                    stream.close()
                    if variant=='sdpa':
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
                            # No update: inspect dispatched forward/backward kernels outside timing.
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                model(**to_device(batch,torch.device('cuda')),use_cache=False).loss.backward()
                            torch.cuda.synchronize()
                        record['attention_operators']=[e.key for e in profiler.key_averages()
                            if 'scaled_dot_product' in e.key or 'flash_attention' in e.key]
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    stream=make_batches(rows,tok,'encoded-prefetch',BATCH,pin_memory=True)
                    n=source=target=padded=0
                    for batch in stream:
                        n+=len(batch['labels']);source+=int(batch['attention_mask'].sum())
                        target+=int(batch['labels'].ne(-100).sum())
                        padded+=batch['input_ids'].numel()+batch['labels'].numel()
                        last=update(batch)
                    torch.cuda.synchronize();seconds=time.perf_counter()-started
                    assert dict(examples=n,source_tokens=source,target_tokens=target,padded_tokens=padded)==ready['counts']
                    assert torch.isfinite(last)
                    record.update(status='complete',seconds=seconds,examples=n,examples_per_second=n/seconds,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,final_loss=float(last))
                except RuntimeError as error:
                    if variant=='native':raise
                    record.update(status='unsupported_or_failed',error=str(error))
                finally:
                    if stream is not None:stream.close()
                    del model,optimizer,stream,batch,last
                    gc.collect();torch.cuda.empty_cache()
                result['runs'].append(record)
                (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(record),flush=True)
        speeds={v:statistics.median(r['examples_per_second'] for r in result['runs'] if r['variant']==v)
            for v in ['native','sdpa'] if sum(r['variant']==v and r['status']=='complete' for r in result['runs'])==2}
        candidate='sdpa' if 'sdpa' in speeds and speeds['sdpa']>=1.05*speeds['native'] else 'native'
        result.update(completed=True,elapsed_seconds=time.monotonic()-began,median_examples_per_second=speeds,
            speed_candidate=candidate,quality_pilot_required=candidate=='sdpa',
            limitation='Short single-GPU throughput probe, same original rows and batch membership. Dropout uses kernel-specific random masks. No trained checkpoint or score improvement is claimed.')
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',flush=True)
    finally:
        monitor.terminate();monitor.wait();telemetry.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    for key in ['pool','tokenizer','gate','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.prepare:cpu_gate(a.pool,a.tokenizer,a.gate)
    else:benchmark(a.pool,a.tokenizer,a.gate,a.output)
