from pathlib import Path
import hashlib,json,time
import torch,pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
from homeworks.hw2.train import new_model,encode_batch,nonembedding_hash
from homeworks.hw2.optimizers import make_optimizer,partition_parameters
from homeworks.hw2.optimizer_pilot import PLAN,LAUNCH
out=Path('homeworks/hw2/experiments/optimizer-scratch-2k-v1');out.mkdir(exist_ok=True)
(out/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
(out/'launch.py').write_text(LAUNCH);compile(LAUNCH,'launch.py','exec')
tokenizer=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer',local_files_only=True)
assert len(tokenizer)==8192
shard=Path('homeworks/hw2/cache/part-00000.parquet')
assert hashlib.sha256(shard.read_bytes()).hexdigest()=='91831d582faa12d73b1a6b3878e496446cd5df074f17ee4ffea8cc2887c708ec'
rows=pq.read_table(shard,columns=['question','code']).to_pylist();excluded={r['question'] for r in rows[:1024]}
examples=[r for r in rows[1024:] if r['question'] not in excluded][:4]
torch.set_num_threads(4);results={}
for arm in ['control','adamw_high','muon_low','muon_high']:
 torch.manual_seed(PLAN['seed'])
 model=new_model(tokenizer,**{n:PLAN[n] for n in ['d_model','d_ff','encoder_layers','decoder_layers','num_heads','initialization_vocab_size']})
 torch.manual_seed(PLAN['seed']);initial=nonembedding_hash(model)
 options=PLAN['arm_options'][arm];opt=make_optimizer(model,options['optimizer'],learning_rate=options['learning_rate'],muon_learning_rate=options.get('muon_learning_rate'))
 batch=encode_batch(examples,tokenizer,torch.device('cpu'));labels=batch['labels'].clone();start=time.monotonic()
 loss=model(**batch,use_cache=False).loss;loss.backward()
 assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 assert all(torch.isfinite(p).all() for p in model.parameters()) and torch.equal(labels,batch['labels'])
 matrices,other,names=partition_parameters(model)
 results[arm]=dict(initial_body_sha256=initial,loss=float(loss.detach()),seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),all_parameters_trainable=all(p.requires_grad for p in model.parameters()),all_parameters_received_finite_gradients=True,post_step_parameters_finite=True,labels_unchanged=True,hidden_matrix_parameters=sum(p.numel() for p in matrices),adamw_embedding_vector_parameters=sum(p.numel() for p in other))
 del model,opt
assert len({v['initial_body_sha256'] for v in results.values()})==1
assert len({v['loss'] for v in results.values()})==1
configs={
 'small':dict(d_model=384,d_ff=1152,encoder_layers=4,decoder_layers=4,num_heads=6),
 'baseline':dict(d_model=512,d_ff=1536,encoder_layers=6,decoder_layers=4,num_heads=8),
 'encoder_heavy':dict(d_model=512,d_ff=1664,encoder_layers=8,decoder_layers=2,num_heads=8),
 'decoder_heavy':dict(d_model=512,d_ff=1408,encoder_layers=4,decoder_layers=6,num_heads=8),
 'large':dict(d_model=640,d_ff=2048,encoder_layers=8,decoder_layers=4,num_heads=10)}
for name,c in configs.items():
 with torch.device('meta'):m=new_model(tokenizer,**c)
 c['parameters']=sum(p.numel() for p in m.parameters())
 del m
record=dict(cpu_verified=True,torch_version=torch.__version__,results=results,scope='Four discarded one-step CPU preflights on original training examples; no saved model or accuracy experiment. Actual pilot not started.',architecture_candidates=configs)
(out/'cpu-preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
