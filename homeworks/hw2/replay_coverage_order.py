"""Independently reconstruct one pass from verified row IDs and cached token lengths.

Does not import the production sampler, tokenize text, train a model, or use GPU.
The cached lengths came from an earlier CPU audit; matching live checkpoint
hashes and token totals tests their compatibility with the current stream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rng(seed,*parts):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,*parts])))


def compare(expected,record):
    state=record['state']
    for key in ['presentations','row_order_sha256','untruncated_tokens','cursor','counts']:
        if expected[key]!=state[key]:raise ValueError(f'Independent replay mismatch: {key}')
    if expected['step']!=record['optimizer_step']:raise ValueError('Wrong checkpoint step')
    return True


def replay(run,index,old_index,lengths,output):
    began=time.monotonic();ready=read(run/'ready.json');plan=ready['plan'];opts=plan['options']
    parent=read(run.parent/'matched-curve-v1/adamw42/step-006018/run_config.json')
    assert np.__version__==ready['runtime_versions']['numpy']
    assert opts['untruncated_training'] and not parent['shuffle_batch_order']
    assert opts['accumulation']==1 and opts['batch_size']==128 and parent['group_window']==32
    assert sha(index/'manifest.json')==ready['fresh_preparation']['index_manifest_sha256']
    manifest=read(index/'manifest.json');old_manifest=read(old_index/'manifest.json')
    tokenizer_hash=read(run/'step-010000/checkpoint.complete.json')['files']['tokenizer.json']
    permutations=[];q_lengths=[];c_lengths=[];eligible=[];seen=[];length_hashes={}
    for s,rec in enumerate(manifest['shards']):
        old_rec=old_manifest['shards'][s];proof=read(lengths/f'shard-{s:05d}.json')
        assert rec['shard']==s and proof['record']==old_rec
        assert proof['identity']['index_manifest_sha256']==sha(old_index/'manifest.json')
        assert proof['identity']['tokenizer_sha256']==tokenizer_hash
        assert rec['parquet_sha256']==old_rec['parquet_sha256']==ready['dataset']['parquet_files'][rec['filename']]
        assert sha(index/f'{s:05d}.npy')==rec['index_sha256']
        assert sha(old_index/f'{s:05d}.npy')==old_rec['index_sha256']
        length_path=lengths/f'shard-{s:05d}.npz';length_hashes[str(s)]=sha(length_path)
        assert length_hashes[str(s)]==proof['length_file_sha256']
        ids=np.load(index/f'{s:05d}.npy',allow_pickle=False)
        old_ids=np.load(old_index/f'{s:05d}.npy',allow_pickle=False)
        assert len(ids)==rec['eligible'] and np.all(ids[1:]>ids[:-1])
        assert len(old_ids)==old_rec['eligible'] and np.all(old_ids[1:]>old_ids[:-1])
        positions=np.searchsorted(old_ids,ids);assert np.array_equal(old_ids[positions],ids)
        with np.load(length_path,allow_pickle=False) as a:
            assert len(a['question'])==len(a['code'])==len(old_ids)
            q=np.zeros(rec['rows'],dtype=np.int64);q[old_ids]=a['question']
            c=np.zeros(rec['rows'],dtype=np.int64);c[old_ids]=a['code']
        q_lengths.append(q);c_lengths.append(c);eligible.append(ids)
        permutations.append(rng(opts['seed'],0,0,s).permutation(ids))
        seen.append(np.zeros(rec['rows'],dtype=bool))
    counts=[0]*len(eligible);total=0;tokens={'question':0,'code':0};chain='0'*64;step=0
    chunk=opts['coverage_rows_per_shard'];batch_size=opts['batch_size'];width=batch_size*parent['group_window']
    rounds=(max(map(len,eligible))+chunk-1)//chunk;checks={};compared=[]
    checkpoints=set(plan['checkpoint_steps']+[plan['parent_step']])
    for r in range(rounds):
        parts=[np.column_stack((np.full(len(a[r*chunk:(r+1)*chunk]),s,dtype=np.int64),a[r*chunk:(r+1)*chunk])) for s,a in enumerate(permutations)]
        rows=np.concatenate(parts);rows=rows[rng(opts['seed'],1,0,r).permutation(len(rows))]
        q=np.empty(len(rows),dtype=np.int64);c=np.empty(len(rows),dtype=np.int64)
        for s in range(len(eligible)):
            mask=rows[:,0]==s;ids=rows[mask,1];assert not seen[s][ids].any()
            assert len(np.unique(ids))==len(ids);seen[s][ids]=True
            q[mask]=q_lengths[s][ids];c[mask]=c_lengths[s][ids]
        batch_number=0
        for start in range(0,len(rows),width):
            end=min(start+width,len(rows));pos=np.arange(start,end)
            # Explicit stable tie-breaker reproduces Python's stable length sort.
            order=pos[np.lexsort((pos,np.minimum(c[pos],192),np.minimum(q[pos],128)))]
            for b in range(0,len(order),batch_size):
                ix=order[b:b+batch_size];ids=rows[ix].astype('<i8',copy=False)
                step+=1;batch_number+=1;total+=len(ids)
                for s,n in zip(*np.unique(ids[:,0],return_counts=True)):counts[int(s)]+=int(n)
                tokens['question']+=int(q[ix].sum());tokens['code']+=int(c[ix].sum())
                chain=hashlib.sha256(bytes.fromhex(chain)+ids.tobytes()).hexdigest()
                cursor=[0,r,batch_number]
                if b+batch_size>=len(order) and end==len(rows):
                    cursor=[0,r+1,0] if r+1<rounds else [1,0,0]
                if step in checkpoints:
                    record=dict(step=step,presentations=total,row_order_sha256=chain,
                        untruncated_tokens=dict(tokens),cursor=cursor,
                        counts=list(counts) if cursor[0]==0 else [0]*len(counts))
                    path=(run.parent/'matched-curve-v1/adamw42/step-006018' if step==plan['parent_step'] else run/f'step-{step:06d}')
                    if (path/'coverage.json').exists():
                        assert sha(path/'coverage.json')==read(path/'checkpoint.complete.json')['files']['coverage.json']
                        compare(record,read(path/'coverage.json'));compared.append(step)
                    checks[str(step)]=record
        print('REPLAY_ROUND',r+1,rounds,'rows',total,'matched_checkpoints',compared,flush=True)
    assert step==plan['final_step'] and total==plan['eligible_rows']
    for s,ids in enumerate(eligible):assert seen[s].sum()==len(ids) and seen[s][ids].all()
    result=dict(numpy_version=np.__version__,source_commit=ready['source_commit'],
        replay_source_sha256=sha(__file__),index_manifest_sha256=sha(index/'manifest.json'),
        tokenizer_sha256=tokenizer_hash,length_file_sha256=length_hashes,
        eligible_rows=total,unique_planned_rows=total,repeated_planned_rows=0,
        compared_checkpoint_steps=compared,checkpoints=checks,elapsed_seconds=time.monotonic()-began,
        gpu_used=False,training_changed=False,
        scope='Independent reconstruction of original row IDs, stable length grouping, per-update hash chain and full-token sums. Compared checkpoints are actual saved consumption ledgers; future checkpoints remain predictions until compared.',
        limitations=['Uses checksum-verified cached lengths from tokenizers 0.21.2; exact comparisons test compatibility with the live stream using tokenizers 0.22.2.',
                    'Does not independently prove gradient computation or optimizer updates; those rely on the tested training consumer.'])
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();
    for key in ['run','index','old-index','lengths','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=replay(a.run,a.index,a.old_index,a.lengths,a.output)
    print('REPLAY_VERIFIED',r['compared_checkpoint_steps'],r['unique_planned_rows'])
