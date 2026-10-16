"""CPU-only corpus-length budget check and real-token preservation probe."""
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast

from homeworks.hw2.checkpoint_io import sha256
from homeworks.hw2.full_coverage import FullCoverage, load_index
from homeworks.hw2.input_pipeline import collate_tokens
from homeworks.hw2.long_sequences import split_batch

started=time.monotonic()
base=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage')
audit=Path('/Users/andrejvorsin/Downloads/hw2-corpus-token-audit-v1')
output=Path('homeworks/hw2/experiments/untruncated-cpu-v1')
manifest,paths,indices=load_index(base/'dataset',base/'index')
tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
kw=dict(token_budget=40960,attention_budget=9961472)
rows=[]; census=0; longest_q=longest_c=max_cost=0
for shard,(path,ids) in enumerate(zip(paths,indices)):
    meta=json.loads((audit/f'shard-{shard:05d}.json').read_text())
    assert meta['record']==manifest['shards'][shard]
    assert meta['identity']['tokenizer_sha256']==sha256(Path('homeworks/hw2/tokenizer/tokenizer.json'))
    lengths_path=audit/f'shard-{shard:05d}.npz'
    assert sha256(lengths_path)==meta['length_file_sha256']
    with np.load(lengths_path,allow_pickle=False) as z:
        q=z['question'].astype(np.int64);c=z['code'].astype(np.int64)
    assert len(q)==len(c)==len(ids)
    cost=q*q+c*c+q*c
    assert np.all(q+c<=kw['token_budget']) and np.all(cost<=kw['attention_budget'])
    census+=len(q);longest_q=max(longest_q,int(q.max()));longest_c=max(longest_c,int(c.max()))
    max_cost=max(max_cost,int(cost.max()))
    chosen=sorted({int(q.argmax()),int(c.argmax()),int(cost.argmax())})
    selected=pq.read_table(path,columns=['question','code']).take(ids[chosen]).to_pylist()
    for pos,row in zip(chosen,selected):
        qi=tok(row['question'],truncation=False)['input_ids']
        ci=tok(row['code'],truncation=False)['input_ids']
        assert len(qi)==q[pos] and len(ci)==c[pos]
        rows.append((qi,ci))
batch=collate_tokens([r[0] for r in rows],[r[1] for r in rows],pad_id=tok.pad_token_id,eos_id=tok.eos_token_id)
chunks=list(split_batch(batch,**kw))
restored=[]
for part,_ in chunks:
    for q,m,c in zip(part['input_ids'],part['attention_mask'],part['labels']):
        restored.append((q[m.bool()].tolist(),c[c!=-100].tolist()))
assert restored==rows
common=dict(manifest=manifest,paths=paths,indices=indices,tokenizer=tok,seed=42,batch_size=128,window=32)
short=FullCoverage(**common);full=FullCoverage(**common,untruncated=True)
a,b=short.batches(),full.batches();tokens=[0,0];chunks_count=0
for _ in range(32):
    capped,complete=next(a),next(b)
    assert capped['_coverage']=={k:v for k,v in complete['_coverage'].items() if k!='untruncated_tokens'}
    ticket=complete.pop('_coverage')
    tokens[0]+=int(complete['attention_mask'].sum());tokens[1]+=int(complete['labels'].ne(-100).sum())
    pieces=list(split_batch(complete,**kw));chunks_count+=len(pieces)
    assert sum(len(x['labels']) for x,_ in pieces)==128
    assert abs(sum(f for _,f in pieces)-1)<1e-12
    full.commit(ticket);short.commit(capped['_coverage'])
assert full.state['row_order_sha256']==short.state['row_order_sha256']
result=dict(cpu_only=True,training_started=False,eligible_rows_budget_checked=census,
    longest_question_tokens=longest_q,longest_program_tokens=longest_c,
    max_single_row_attention_positions=max_cost,budgets=kw,
    extreme_original_examples_verified=len(rows),extreme_microbatches=len(chunks),
    regular_window_rows=4096,regular_window_microbatches=chunks_count,
    untruncated_tokens=tokens,row_order_sha256=full.state['row_order_sha256'],
    complete_token_ids_preserved=True,consumed_token_receipt=full.state['untruncated_tokens'],full_training_pass_completed=False,
    seconds=time.monotonic()-started,
    limitations=['Budgets are proxies, not measured GPU memory; GPU throughput probe still required.',
                 'Full corpus lengths checked; token preservation retested on selected extremes and first window only.'])
(output/'real-data-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
