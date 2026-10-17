from pathlib import Path
import json,time,gc
import torch
import pyarrow.parquet as pq
from homeworks.hw2.train import new_model,parameter_hash
from homeworks.hw2.digit_tokenization import load_training_tokenizer
from homeworks.hw2.adamw_size_curve import PLAN,LAUNCH
OPTIONS=PLAN["options"]; CONFIGS=PLAN["arms"]
from homeworks.hw2.full_coverage import FullCoverage,load_index
from homeworks.hw2.long_sequences import split_batch
from homeworks.hw2.optimizers import make_optimizer
from homeworks.hw2.input_pipeline import collate_tokens
from homeworks.hw2.checkpoint_io import sha256

assert not torch.cuda.is_available();torch.set_num_threads(2)
e=Path('homeworks/hw2/experiments/adamw-size-curve-v1');e.mkdir(exist_ok=True)
(e/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');(e/'launch.py').write_text(LAUNCH)
cache=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset')
index=Path('/Users/andrejvorsin/Downloads/hw2-fresh-holdout-v2-restored/index')
m,paths,ids=load_index(cache,index)
tok=load_training_tokenizer(Path('homeworks/hw2/tokenizer'))
stream=FullCoverage(m,paths,ids,tok,seed=42,batch_size=128,window=32,rows_per_shard=4096,untruncated=True)
assert sum(stream.batches_in_round(r) for r in range(stream.rounds))==OPTIONS['steps']
batch=next(stream.batches());ticket=batch.pop('_coverage')
# Audit original strings at the actual first consumed row IDs.
rows=[]
for shard,row in ticket['row_ids'][:4]:
 rows.append(pq.read_table(paths[shard],columns=['question','code']).slice(row,1).to_pylist()[0])
source=tok([r['question'] for r in rows],truncation=False)['input_ids']
target=tok([r['code'] for r in rows],truncation=False)['input_ids']
tiny=collate_tokens(source,target,pad_id=tok.pad_token_id,eos_id=tok.eos_token_id)
results={};hashes={}
for name,config in CONFIGS.items():
 torch.manual_seed(42);began=time.monotonic()
 model=new_model(tok,encoder_layers=6,decoder_layers=4,initialization_vocab_size=16384,
                 **{k:v for k,v in config.items() if k in ('d_model','d_ff','num_heads')})
 params=sum(p.numel() for p in model.parameters());assert params==78146112
 hashes[name]=parameter_hash(model)
 optimizer=make_optimizer(model,config['optimizer'],learning_rate=.0003,muon_learning_rate=config.get('muon_learning_rate'))
 loss=model(**tiny,use_cache=False).loss;loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
 model.eval()
 with torch.inference_mode():
  kwargs={k:v[:1] for k,v in tiny.items() if k!='labels'}
  cached=model.generate(**kwargs,max_new_tokens=12,do_sample=False,num_beams=1,use_cache=True)
  plain=model.generate(**kwargs,max_new_tokens=12,do_sample=False,num_beams=1,use_cache=False)
 assert torch.equal(cached,plain)
 results[name]=dict(parameters=params,loss=float(loss.detach()),gradient_norm=float(norm),
    finite_after_update=True,cached_greedy_equal=True,seconds=time.monotonic()-began)
 (e/'cpu-preflight-progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,'CPU update verified',flush=True)
 del model,optimizer,loss;gc.collect()
chunks=list(split_batch(batch,token_budget=40960,attention_budget=9961472))
assert sum(b['labels'].shape[0] for b,_ in chunks)==128
assert sum(int(b['labels'].ne(-100).sum()) for b,_ in chunks)==ticket['untruncated_tokens']['code']
assert sum(int(b['attention_mask'].sum()) for b,_ in chunks)==ticket['untruncated_tokens']['question']
result=dict(cpu_verified=True,arms=results,initial_model_hashes=hashes,gpu_initialization_verified=False,
 fresh_index_sha256=sha256(index/'manifest.json'),eligible_rows=sum(len(x) for x in ids),
 first_batch_complete_tokens=True,first_batch_microbatches=len(chunks),full_pass_updates=OPTIONS['steps'],
 original_targets_unchanged=True,training_started=False)
(e/'cpu-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
