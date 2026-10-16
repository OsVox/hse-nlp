"""Verify the real full-pass row plan and measure CPU input preparation only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from transformers import PreTrainedTokenizerFast

from .checkpoint_io import sha256
from .data import training_batches
from .full_coverage import FullCoverage, load_index, prepare_index
from .input_pipeline import encoded_batches


def run(cache, index, output, benchmark_batches=128):
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if not (index/'manifest.json').exists():
        prepare_index(cache,index)
    manifest,paths,indices=load_index(cache,index)
    index_seconds=time.monotonic()-started
    tokenizer=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    stream=FullCoverage(manifest,paths,indices,tokenizer,seed=42,batch_size=128)
    seen=[np.zeros(r['rows'],dtype=bool) for r in manifest['shards']]
    permutations=stream.epoch_permutations(0)
    order_digest=hashlib.sha256()
    rounds=[]
    started=time.monotonic()
    for round_id in range(stream.rounds):
        plan=stream.round_plan(0,round_id,permutations)
        counts=[]
        for shard,bitmap in enumerate(seen):
            ids=plan[plan[:,0]==shard,1]
            if len(np.unique(ids)) != len(ids) or bitmap[ids].any():
                raise AssertionError('Repeated original row before end of full pass')
            bitmap[ids]=True
            counts.append(len(ids))
        order_digest.update(plan.astype('<i8').tobytes())
        rounds.append(dict(round=round_id,rows=len(plan),per_shard=counts))
    for bitmap,eligible in zip(seen,indices):
        if not np.array_equal(np.flatnonzero(bitmap),eligible):
            raise AssertionError('Missing eligible row or included held-out row')
    plan_seconds=time.monotonic()-started
    result=dict(index_seconds_including_hash_checks=index_seconds,
        row_plan_seconds=plan_seconds,eligible_rows=sum(map(len,indices)),
        exact_row_plan_visits=sum(r['rows'] for r in rounds),
        batches_per_full_pass_at_128=sum(stream.batches_in_round(i) for i in range(stream.rounds)),
        repeated_row_ids=0,missing_eligible_rows=0,heldout_rows_visited=0,
        rounds=rounds,pre_grouping_row_order_sha256=order_digest.hexdigest(),
        index_manifest_sha256=sha256(index/'manifest.json'),
        source_sha256=sha256(Path(__file__).with_name('full_coverage.py')),
        limitation='Row-plan traversal is a CPU verification, not model training. Unit tests cover grouping and consumed-state resume.',
        gpu_used=False,benchmarks=[])
    # Alternating order records both startup and warm-cache effects explicitly.
    for mode in ['coverage','legacy','legacy','coverage']:
        tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
        if mode=='coverage':
            sampler=FullCoverage(manifest,paths,indices,tok,seed=42,batch_size=128)
            batches=sampler.batches()
        else:
            batches=encoded_batches(training_batches(cache,128,seed=42,
                mixed_rows_per_shard=4096),tok,128,32)
        started=time.monotonic()
        rows=0
        first=None
        for _ in range(benchmark_batches):
            batch=next(batches)
            rows+=len(batch['labels'])
            if first is None:
                first=time.monotonic()-started
        seconds=time.monotonic()-started
        batches.close()
        result['benchmarks'].append(dict(mode=mode,batches=benchmark_batches,
            rows=rows,seconds=seconds,first_batch_seconds=first,rows_per_second=rows/seconds))
        print(json.dumps(result['benchmarks'][-1]),flush=True)
    result['benchmark_limitation']='Local CPU input preparation; different row orders/lengths, warm caches, no CUDA or end-to-end speed claim.'
    (output/'real-data-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['eligible_rows','exact_row_plan_visits',
        'repeated_row_ids','missing_eligible_rows','heldout_rows_visited','gpu_used']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--index-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--benchmark-batches',type=int,default=128)
    args=parser.parse_args()
    if args.benchmark_batches<1:
        parser.error('Benchmark needs at least one batch')
    run(args.cache_dir,args.index_dir,args.output_dir,args.benchmark_batches)
