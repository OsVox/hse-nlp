from pathlib import Path
import json,time
import torch
from transformers import PreTrainedTokenizerFast,T5ForConditionalGeneration
from homeworks.hw2.complete_curve import PLAN,LAUNCH
from homeworks.hw2.checkpoint_io import sha256
from homeworks.hw2.full_coverage import FullCoverage,load_index
from homeworks.hw2.input_pipeline import collate_tokens
from homeworks.hw2.long_sequences import split_batch
from homeworks.hw2.token_weighting import token_weights,weighted_loss
out=Path('homeworks/hw2/experiments/complete-curve-4k-v1')
base=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage');parent=Path('/Users/andrejvorsin/Downloads/hw2-frequency-screen/control/step-073024')
assert not torch.cuda.is_available();torch.set_num_threads(2);began=time.monotonic()
marker=json.loads((parent/'checkpoint.complete.json').read_text())
for name,digest in marker['files'].items():assert sha256(parent/name)==digest
assert sha256(parent/'model.safetensors')==PLAN['parent_model_sha256']
manifest,paths,indices=load_index(base/'dataset',base/'index')
tok=PreTrainedTokenizerFast.from_pretrained(parent)
s=FullCoverage(manifest,paths,indices,tok,seed=42,batch_size=128,window=32,untruncated=True,start_step=PLAN['parent_step'])
it=s.batches();micro=0
for _ in range(32):
 b=next(it);micro+=len(list(split_batch({k:v for k,v in b.items() if not k.startswith("_")},token_budget=40960,attention_budget=9961472)))
 s.commit(b['_coverage'])
it.close()
assert s.state['presentations']==4096
pool=Path('/tmp/untruncated-gpu-probe-v1/pool.json')
assert sha256(pool)=='e34e56375e18c763206629e71d6d7cf70eaf0eca3b4e590a3c6d34266f9b1d36'
rows=json.loads(pool.read_text())['extreme']
encoded=[(tok(r['question'])['input_ids'],tok(r['code'])['input_ids']) for r in rows]
chosen=[max(encoded,key=lambda x:len(x[0])),max(encoded,key=lambda x:len(x[1]))]
batch=collate_tokens([r[0] for r in chosen],[r[1] for r in chosen],pad_id=tok.pad_token_id,eos_id=tok.eos_token_id)
model=T5ForConditionalGeneration.from_pretrained(parent,local_files_only=True);model.train()
state=torch.load(parent/'training_state.pt',weights_only=False,map_location='cpu');assert state['step']==PLAN['parent_step']
optimizer=torch.optim.AdamW(model.parameters(),lr=PLAN['learning_rate']);optimizer.load_state_dict(state['optimizer'])
for parameter,values in optimizer.state.items():
 assert values['exp_avg'].shape==parameter.shape==values['exp_avg_sq'].shape
 assert torch.isfinite(values['exp_avg']).all() and torch.isfinite(values['exp_avg_sq']).all()
weights=token_weights(tok,2.)
loss=weighted_loss(model(**batch,use_cache=False).logits,batch['labels'],weights);loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
record=dict(cpu_verified=True,parent_model_sha256=PLAN['parent_model_sha256'],parent_step=PLAN['parent_step'],optimizer_state_verified=True,eligible_rows=sum(map(len,indices)),tracked_window_receipt=s.checkpoint(),logical_batches=32,microbatches=micro,actual_parent_long_examples=2,question_shape=list(batch['input_ids'].shape),program_shape=list(batch['labels'].shape),numeric_weighted_loss=float(loss.detach()),seconds=time.monotonic()-began,weights_saved=False,optimizer_update_performed=False)
(out/'cpu-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
(out/'runtime-plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');(out/'launch.py').write_text(LAUNCH)
compile(LAUNCH,'launch.py','exec');print(json.dumps({k:v for k,v in record.items() if k!='tracked_window_receipt'},indent=2))
