from pathlib import Path
import json,time,gc
import torch
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
from homeworks.hw2.train import new_model,encode_batch,parameter_hash,nonembedding_hash
from homeworks.hw2.optimizers import make_optimizer
from homeworks.hw2.activation_pilot import PLAN,LAUNCH,ACTIVATIONS
from homeworks.hw2.checkpoint_io import sha256
from homeworks.hw2.full_coverage import load_index
root=Path('homeworks/hw2');e=root/'experiments/activation-muon-2k-v1';e.mkdir(exist_ok=True)
assert not torch.cuda.is_available();torch.set_num_threads(2);began=time.monotonic()
base=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage');manifest,paths,indices=load_index(base/'dataset',base/'index')
tok=PreTrainedTokenizerFast.from_pretrained(root/'tokenizer');table=pq.read_table(paths[0],columns=['question','code']);examples=table.take(indices[0][:4]).to_pylist();batch=encode_batch(examples,tok,torch.device('cpu'))
records={}
for arm,activation in ACTIVATIONS.items():
 torch.manual_seed(42)
 m=new_model(tok,**{k:PLAN[k] for k in ['d_model','d_ff','encoder_layers','decoder_layers','num_heads','initialization_vocab_size']},feed_forward_proj=activation)
 count=sum(p.numel() for p in m.parameters());assert count==PLAN['expected_parameters'][arm]
 initial=parameter_hash(m);body=nonembedding_hash(m);torch.manual_seed(42)
 opt=make_optimizer(m,'muon',learning_rate=.0003,muon_learning_rate=.001);m.train();loss=m(**batch,use_cache=False).loss;loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();after=parameter_hash(m)
 assert after!=initial and all(torch.isfinite(p).all() for p in m.parameters())
 records[arm]=dict(parameters=count,initial_model_sha256=initial,initial_body_sha256=body,activation=m.config.feed_forward_proj,loss=float(loss.detach()),gradient_norm=norm,finite_after_update=True)
 del m,opt;gc.collect()
assert records['control']['initial_model_sha256']==records['swiglu']['initial_model_sha256']
r=dict(cpu_verified=True,results=records,actual_original_examples=4,all_initial_parameters_identical=True,torch_version=torch.__version__,gpu_used=False,weights_saved=False,eligible_rows=sum(map(len,indices)),all_shards_and_indices_hash_verified=True,coverage_manifest_sha256=sha256(base/'index/manifest.json'),elapsed_seconds=time.monotonic()-began)
(e/'cpu-preflight.json').write_text(json.dumps(r,indent=2)+'\n');(e/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');(e/'launch.py').write_text(LAUNCH);compile(LAUNCH,'launch.py','exec');print(json.dumps(r,indent=2))
