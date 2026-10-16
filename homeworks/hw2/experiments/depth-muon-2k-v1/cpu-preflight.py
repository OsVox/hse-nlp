from pathlib import Path
import json,hashlib,time,gc
import torch,pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
from homeworks.hw2.train import new_model,encode_batch,nonembedding_hash
from homeworks.hw2.optimizers import partition_parameters
from homeworks.hw2.depth_pilot import PLAN,LAUNCH,CONFIGS
out=Path('homeworks/hw2/experiments/depth-muon-2k-v1')
(out/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');(out/'launch.py').write_text(LAUNCH);compile(LAUNCH,'launch.py','exec')
t=torch; t.set_num_threads(4)
tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer',local_files_only=True)
p=Path('homeworks/hw2/cache/part-00000.parquet');assert hashlib.sha256(p.read_bytes()).hexdigest()=='91831d582faa12d73b1a6b3878e496446cd5df074f17ee4ffea8cc2887c708ec'
rows=pq.read_table(p,columns=['question','code']).to_pylist();excluded={r['question'] for r in rows[:1024]};examples=[r for r in rows[1024:] if r['question'] not in excluded][:2]
batch=encode_batch(examples,tok,t.device('cpu'));labels=batch['labels'].clone();results={}
for name,c in CONFIGS.items():
 t.manual_seed(42);m=new_model(tok,d_model=512,num_heads=8,initialization_vocab_size=16384,**c)
 count=sum(p.numel() for p in m.parameters());assert count==PLAN['expected_parameters'][name]
 initial=nonembedding_hash(m);t.manual_seed(42);start=time.monotonic();loss=m(**batch,use_cache=False).loss;loss.backward()
 assert t.isfinite(loss) and all(p.grad is not None and t.isfinite(p.grad).all() for p in m.parameters())
 assert t.equal(labels,batch['labels'])
 matrices,other,names=partition_parameters(m);assert sum(p.numel() for p in matrices+other)==count
 m.eval()
 with t.no_grad():generated=m.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],num_beams=1,do_sample=False,max_new_tokens=4,use_cache=True)
 assert generated.shape[0]==2 and generated.shape[1]<=5
 results[name]={'parameters':count,'initial_body_sha256':initial,'loss':loss.item(),'seconds':time.monotonic()-start,'all_parameter_gradients_finite':True,'original_labels_unchanged':True,'cached_greedy_generation_passed':True,'muon_parameters':sum(p.numel() for p in matrices),'adamw_parameters':sum(p.numel() for p in other)}
 del m;gc.collect()
record={'cpu_verified':True,'torch_version':t.__version__,'results':results,'scope':'Discarded full-size CPU forward/backward and cached greedy checks on two original examples per architecture. No training checkpoint or accuracy claim. Native optimizer updates were covered by the unchanged optimizer tests and previous full-size preflight.'}
(out/'cpu-preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
