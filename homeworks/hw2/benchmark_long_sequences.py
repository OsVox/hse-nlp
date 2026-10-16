"""Disposable GPU memory/throughput probe; no checkpoint or quality claim."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast

from .checkpoint_io import sha256
from .full_coverage import FullCoverage, load_index
from .input_pipeline import collate_tokens, to_device
from .long_sequences import split_batch
from .optimizers import make_optimizer
from .token_weighting import token_weights, weighted_loss
from .train import new_model

BUDGETS=dict(token_budget=40960,attention_budget=9961472)


def batches(rows,tok,complete):
    q=tok([r['question'] for r in rows],**({'truncation':False} if complete else {'truncation':True,'max_length':128}))['input_ids']
    c=tok([r['code'] for r in rows],**({'truncation':False} if complete else {'truncation':True,'max_length':192}))['input_ids']
    return [collate_tokens(q[i:i+128],c[i:i+128],pad_id=tok.pad_token_id,eos_id=tok.eos_token_id)
            for i in range(0,len(rows),128)]


def prepare(cache,index,lengths,output):
    assert not torch.cuda.is_available()
    output.mkdir(exist_ok=False,parents=True)
    tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    manifest,paths,ids=load_index(cache,index)
    sampler=FullCoverage(manifest,paths,ids,tok,seed=42,batch_size=128,window=32,untruncated=True)
    stream=sampler.batches();regular=[]
    for _ in range(32):
        b=next(stream);regular.extend(b['_coverage']['row_ids']);sampler.commit(b['_coverage'])
    stream.close()
    extreme=[];lookup={}
    for shard,(path,eligible) in enumerate(zip(paths,ids)):
        p=lengths/f'shard-{shard:05d}.npz'
        record=json.loads((lengths/f'shard-{shard:05d}.json').read_text())
        assert record['record']==manifest['shards'][shard] and sha256(p)==record['length_file_sha256']
        with np.load(p,allow_pickle=False) as z:q=z['question'].astype(np.int64);c=z['code'].astype(np.int64)
        chosen=sorted({int(q.argmax()),int(c.argmax()),int((q*q+c*c+q*c).argmax())})
        extreme.extend((shard,int(eligible[i])) for i in chosen)
        selected=sorted({i for s,i in regular+extreme if s==shard})
        for i,row in zip(selected,pq.read_table(path,columns=['question','code']).take(selected).to_pylist()):
            lookup[(shard,i)]={**row,'row_id':[shard,i]}
    pool={name:[lookup[tuple(i)] for i in group] for name,group in [('regular',regular),('extreme',extreme)]}
    p=output/'pool.json';p.write_text(json.dumps(pool,ensure_ascii=False))
    counts={}
    for name,rows in pool.items():
        encoded=batches(rows,tok,True)
        counts[name]=dict(rows=len(rows),microbatches=sum(len(list(split_batch(b,**BUDGETS))) for b in encoded),
                          question_tokens=sum(int(b['attention_mask'].sum()) for b in encoded),
                          program_tokens=sum(int(b['labels'].ne(-100).sum()) for b in encoded))
    gate=dict(pool_sha256=sha256(p),tokenizer_sha256=sha256(Path('homeworks/hw2/tokenizer/tokenizer.json')),
              split_source_sha256=sha256(Path(__file__).with_name('long_sequences.py')),budgets=BUDGETS,
              counts=counts,row_order_sha256=sampler.state['row_order_sha256'],cpu_verified=True,
              weights_discarded=True,training_quality_claim=False)
    (output/'gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate),flush=True)


def benchmark(prepared,output):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    gate=json.loads((prepared/'gate.json').read_text());assert gate['cpu_verified']
    assert sha256(prepared/'pool.json')==gate['pool_sha256']
    assert sha256(Path(__file__).with_name('long_sequences.py'))==gate['split_source_sha256']
    assert sha256(Path('homeworks/hw2/tokenizer/tokenizer.json'))==gate['tokenizer_sha256']
    pool=json.loads((prepared/'pool.json').read_text())
    tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    weight_cpu=token_weights(tok,2.);weights=weight_cpu.cuda()
    output.mkdir(parents=True,exist_ok=False)
    result=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,gate=gate,
                precision='bf16',optimizer='Muon .001 / AdamW .0003',numeric_token_weight=2,
                parameters=42481152,weights_discarded=True,quality_claim=False,runs=[])
    start=time.monotonic();log=(output/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=log)
    try:
        for repeat,order in enumerate([('capped','complete'),('complete','capped')]):
            for variant in order:
                torch.manual_seed(42)
                model=new_model(tok,d_model=512,d_ff=1536,encoder_layers=6,decoder_layers=4,num_heads=8).cuda().train()
                assert sum(p.numel() for p in model.parameters())==result['parameters']
                optimizer=make_optimizer(model,'muon',learning_rate=.0003,muon_learning_rate=.001)
                prep=time.monotonic();encoded=batches(pool['regular'],tok,variant=='complete')
                preparation=time.monotonic()-prep
                def update(batch):
                    optimizer.zero_grad(set_to_none=True)
                    pieces=split_batch(batch,**BUDGETS,weights=weight_cpu) if variant=='complete' else [(batch,1.)]
                    count=0;total=torch.zeros((),device='cuda')
                    for part,fraction in pieces:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            output=model(**to_device(part,torch.device('cuda')),use_cache=False)
                            loss=weighted_loss(output.logits,part['labels'].cuda(),weights)*fraction
                        loss.backward();total+=loss.detach();count+=1
                        del output,loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();return total,count
                for batch in encoded[:3]:last,_=update(batch)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter();chunks=0
                for batch in encoded:
                    last,n=update(batch);chunks+=n
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                record=dict(repeat=repeat,variant=variant,examples=4096,updates=32,microbatches=chunks,
                    seconds=elapsed,examples_per_second=4096/elapsed,cpu_preparation_seconds=preparation,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    final_loss=float(last))
                if variant=='complete' and repeat==0:
                    torch.cuda.reset_peak_memory_stats();stress=time.perf_counter();stress_chunks=0
                    for batch in batches(pool['extreme'],tok,True):
                        last,n=update(batch);stress_chunks+=n
                    torch.cuda.synchronize()
                    record['extreme_probe']=dict(examples=len(pool['extreme']),microbatches=stress_chunks,
                        seconds=time.perf_counter()-stress,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        finite_loss=bool(torch.isfinite(last)))
                result['runs'].append(record);print(json.dumps(record),flush=True)
                (output/'progress.json').write_text(json.dumps(result,indent=2))
                del model,optimizer,encoded,batch,last
                gc.collect();torch.cuda.empty_cache()
        result['driver_seconds']=time.monotonic()-start
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        monitor.terminate();monitor.wait(timeout=10);log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--index',type=Path);p.add_argument('--lengths',type=Path);p.add_argument('--prepared',type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a.cache,a.index,a.lengths,a.output)
    else:benchmark(a.prepared,a.output)
