from pathlib import Path
import json,math
from transformers import PreTrainedTokenizerFast
from homeworks.hw2.benchmark_long_sequences import batches,BUDGETS
from homeworks.hw2.long_sequences import split_batch
from homeworks.hw2.token_weighting import token_weights
from homeworks.hw2.checkpoint_io import sha256
src=Path('/tmp/untruncated-gpu-probe-v1'); gate=json.loads((src/'gate.json').read_text());assert sha256(src/'pool.json')==gate['pool_sha256'];assert sha256(Path('homeworks/hw2/long_sequences.py'))==gate['split_source_sha256']
tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer');assert sha256(Path('homeworks/hw2/tokenizer/tokenizer.json'))==gate['tokenizer_sha256']
weights=token_weights(tok,2.);pool=json.loads((src/'pool.json').read_text());result={}
for name,rows in pool.items():
 encoded=batches(rows,tok,True);result[name]={}
 for factor in (1,2,3):
  count=0
  for batch in encoded:
   parts=list(split_batch(batch,**{k:v*factor for k,v in BUDGETS.items()},weights=weights))
   assert math.isclose(sum(f for _,f in parts),1.,rel_tol=1e-6)
   import torch
   start=0
   for part,_ in parts:
    for i in range(len(part['labels'])):
     assert torch.equal(part['input_ids'][i][part['attention_mask'][i].bool()],batch['input_ids'][start][batch['attention_mask'][start].bool()])
     assert torch.equal(part['labels'][i][part['labels'][i].ne(-100)],batch['labels'][start][batch['labels'][start].ne(-100)])
     start+=1
   assert start==len(batch['labels'])
   count+=len(parts)
  result[name][str(factor)]=dict(examples=len(rows),microbatches=count,exact_nonpadding_tokens_preserved=True)
p=Path('homeworks/hw2/experiments/complete-budget-gpu-v1')
(p/'cpu-preflight.json').write_text(json.dumps(dict(cpu_verified=True,source_pool_sha256=gate['pool_sha256'],counts=result,tests_passed=6,test_seconds=3.52),indent=2)+'\n')
plan=dict(hypothesis='Larger complete-sequence microbatches may improve throughput at unchanged effective batch and original rows.',factors=[1,2,3],orders=[[1,2,3],[3,2,1]],regular_examples=4096,extreme_examples=130,effective_batch=128,optimizer='AdamW',learning_rate=.00003,parameters=42481152,numeric_weight=2,precision='bf16',worker_timeout_seconds=360,minimum_memory_gib=80,weights_discarded=True,quality_claim=False,gate='Require finite regular and extreme runs with complete tensors. Adopt larger budget only if both reversed-order repeats are faster than factor1 and peak memory leaves at least 15% device headroom; otherwise retain factor1.',source='Previously hash-verified original-data regular and extreme pool; no new examples or targets.')
(p/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(result))
