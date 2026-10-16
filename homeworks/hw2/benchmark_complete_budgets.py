"""Disposable complete-sequence microbatch budget comparison on identical rows."""
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from transformers import PreTrainedTokenizerFast

from .benchmark_long_sequences import BUDGETS, batches
from .checkpoint_io import sha256
from .input_pipeline import to_device
from .long_sequences import split_batch
from .optimizers import make_optimizer
from .token_weighting import token_weights, weighted_loss
from .train import new_model


def run(prepared, output):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 80*1024**3
    torch.set_num_threads(4)
    gate = json.loads((prepared/'gate.json').read_text())
    assert gate['cpu_verified'] and sha256(prepared/'pool.json') == gate['pool_sha256']
    assert sha256(Path(__file__).with_name('long_sequences.py')) == gate['split_source_sha256']
    assert sha256(Path('homeworks/hw2/tokenizer/tokenizer.json')) == gate['tokenizer_sha256']
    tok = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    pool = json.loads((prepared/'pool.json').read_text())
    encoded = batches(pool['regular'],tok,True)
    extremes = batches(pool['extreme'],tok,True)
    weight_cpu = token_weights(tok,2.)
    weights = weight_cpu.cuda()
    output.mkdir(parents=True,exist_ok=False)
    result = dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                  effective_batch=128,parameters=42481152,precision='bf16',optimizer='AdamW',
                  base_budgets=BUDGETS,weights_discarded=True,quality_claim=False,
                  timing_scope='Pretokenized batches, transfer, forward/backward, clipping and optimizer steps. Includes complete original sequences; no data-loader speed claim.',runs=[])
    log=(output/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=log)
    started=time.monotonic()
    try:
        for repeat,order in enumerate(((1,2,3),(3,2,1))):
            for factor in order:
                torch.manual_seed(42)
                model=new_model(tok,d_model=512,d_ff=1536,encoder_layers=6,decoder_layers=4,num_heads=8).cuda().train()
                optimizer=make_optimizer(model,'adamw',learning_rate=3e-5)
                budgets={k:v*factor for k,v in BUDGETS.items()}
                def update(batch):
                    optimizer.zero_grad(set_to_none=True)
                    pieces=list(split_batch(batch,**budgets,weights=weight_cpu))
                    assert sum(len(p['labels']) for p,_ in pieces)==len(batch['labels'])
                    assert sum(int(p['labels'].ne(-100).sum()) for p,_ in pieces)==int(batch['labels'].ne(-100).sum())
                    assert math.isclose(sum(f for _,f in pieces),1.,rel_tol=1e-6)
                    total=torch.zeros((),device='cuda')
                    for part,fraction in pieces:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            value=model(**to_device(part,torch.device('cuda')),use_cache=False)
                            loss=weighted_loss(value.logits,part['labels'].cuda(),weights)*fraction
                        loss.backward();total+=loss.detach()
                        del value,loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    return total,len(pieces)
                for batch in encoded[:3]:last,_=update(batch)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter();chunks=0
                for batch in encoded:
                    last,n=update(batch);chunks+=n
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                assert bool(torch.isfinite(last))
                record=dict(repeat=repeat,budget_factor=factor,budgets=budgets,examples=len(pool['regular']),
                            updates=len(encoded),microbatches=chunks,seconds=elapsed,
                            examples_per_second=len(pool['regular'])/elapsed,final_loss=float(last),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                if repeat==0:
                    torch.cuda.reset_peak_memory_stats();stress=time.perf_counter();stress_chunks=0
                    for batch in extremes:
                        last,n=update(batch);stress_chunks+=n
                    torch.cuda.synchronize();assert bool(torch.isfinite(last))
                    record['extreme_probe']=dict(examples=len(pool['extreme']),microbatches=stress_chunks,
                        seconds=time.perf_counter()-stress,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),finite=True)
                result['runs'].append(record)
                (output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                del model,optimizer,batch,last
                gc.collect();torch.cuda.empty_cache()
        result['driver_seconds']=time.monotonic()-started
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        monitor.terminate();monitor.wait(timeout=10);log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
