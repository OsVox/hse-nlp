"""CPU-only fitting of 545 copy parameters; Transformer weights remain frozen."""
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import torch
from transformers import PreTrainedTokenizerFast,T5ForConditionalGeneration

from .checkpoint_io import archive_checkpoint,drive_preflight,package_jsonl,sha256
from .copy_adapter import CopyT5,from_base_checkpoint,normalized_heads
from .data import training_shard,validation_examples
from .inference import generate_predictions
from .numeric_eval import score_programs,save_generation
from .short_pilots import ROOT,copy_verified
from .token_weighting import token_weights
from .train import encode_batch
from .verify_export import verify

NAME='copy-adapter-cpu-v1'
PLAN=json.loads((Path(__file__).parent/'experiments'/NAME/'plan.json').read_text())
RUN=ROOT/'runs'/NAME


def base_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not name.startswith('copy_'):
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def features(model,tokenizer,examples):
    pieces=[];weights=token_weights(tokenizer,2.)
    for start in range(0,len(examples),8):
        b=encode_batch(examples[start:start+8],tokenizer,torch.device('cpu'))
        y=b['labels'];valid=y.ne(-100)
        o=T5ForConditionalGeneration.forward(model,**b,use_cache=False,output_attentions=True,
            output_hidden_states=True,return_dict=True)
        heads,has_source=normalized_heads(o.cross_attentions,b['input_ids'],b['attention_mask'],
            tokenizer.pad_token_id,tokenizer.eos_token_id)
        assert has_source.all()
        match=y[:,:,None].eq(b['input_ids'][:,None,:])
        mass=(heads*match[:,:,None,:]).sum(-1)
        p=o.logits.float().softmax(-1).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        pieces.append((o.decoder_hidden_states[-1][valid].float(),mass[valid],p[valid],weights[y[valid]]))
        if (start+8)%256==0:print('FEATURE_ROWS',min(start+8,len(examples)),len(examples),flush=True)
    # Return ordinary tensors: fitting needs autograd on the adapter parameters.
    with torch.inference_mode(False):
        return tuple(torch.cat([x[i] for x in pieces]).clone() for i in range(4))


def likelihood(model,data,indices):
    hidden,mass,p,weight=(x[indices] for x in data)
    gate=model.copy_probability(hidden).squeeze(-1)
    copy=(mass*model.copy_head_logits.softmax(0)).sum(-1)
    prob=((1-gate)*p+gate*copy).clamp_min(torch.finfo(torch.float32).tiny)
    return -(prob.log()*weight).sum()/weight.sum()


@torch.inference_mode()
def evaluate(model,tokenizer,examples,output,offset):
    answers=[];ids_all=[];began=time.monotonic()
    for start in range(0,len(examples),16):
        items=examples[start:start+16]
        b=tokenizer([x['question'] for x in items],padding=True,truncation=True,max_length=128,
            return_tensors='pt',return_token_type_ids=False)
        ids=model.generate(**b,max_new_tokens=192,num_beams=1,do_sample=False,use_cache=True)
        ids_all.extend(ids.tolist())
        answers.extend(tokenizer.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
    metadata=dict(offset=offset,rows=len(examples),selection='greedy',num_beams=1,
        copy_enabled=model.copy_enabled,generation_seconds=time.monotonic()-began,
        selected_token_ids=ids_all,device='cpu',precision='float32',handwritten_repairs=0)
    save_generation(output,answers,[x['code'] for x in examples],metadata)
    result=score_programs(answers,[x['code'] for x in examples])
    result.update({k:v for k,v in metadata.items() if k!='selected_token_ids'})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('HELDOUT',output.parent.name,output.stem,result['numeric_correct'],result['prediction_failures'],flush=True)
    return result


def run():
    import pyarrow.parquet as pq
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    torch.set_num_threads(4);torch.manual_seed(PLAN['seed'])
    drive_preflight(ROOT);assert not RUN.exists()
    began=time.monotonic();local=Path('/content/copy-adapter-work');local.mkdir(exist_ok=False)
    parent=local/'parent';parent.mkdir()
    source=Path(PLAN['parent_mount_path'])
    manifest=json.loads((source/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==PLAN['parent_model_sha256']
    for name,digest in manifest['files'].items():
        assert Path(name).name==name
        if name!='training_state.pt':
            copy_verified(source/name,parent/name);assert sha256(parent/name)==digest
    tokenizer=PreTrainedTokenizerFast.from_pretrained(parent,local_files_only=True)
    model=from_base_checkpoint(parent).float().cpu().eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==545
    before=base_hash(model)
    cache=local/'cache';heldout=validation_examples(cache)
    assert sha256(cache/'part-00000.parquet')==PLAN['train_shard_sha256']
    excluded={r['question'] for r in heldout}
    rows=pq.read_table(training_shard(0,cache),columns=['question','code']).to_pylist()
    eligible=[i for i,r in enumerate(rows) if i>=1024 and r['question'] not in excluded]
    indices=random.Random(PLAN['seed']).sample(eligible,PLAN['training_rows'])
    examples=[rows[i] for i in indices]
    train_features=features(model,tokenizer,examples)
    fitting=time.monotonic()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=PLAN['learning_rate'])
    generator=torch.Generator().manual_seed(PLAN['seed'])
    history=[]
    for step in range(1,PLAN['adapter_updates']+1):
        selected=torch.randint(len(train_features[0]),(PLAN['token_batch_size'],),generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss=likelihood(model,train_features,selected)
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step%64==0:
            record=dict(step=step,loss=float(loss.detach()));history.append(record);print('ADAPTER_STEP',record,flush=True)
    assert base_hash(model)==before
    fit_seconds=time.monotonic()-fitting;del train_features
    RUN.mkdir(parents=True)
    config=dict(plan=PLAN,source_commit=Path('source_commit.txt').read_text().strip(),
        base_parameter_sha256=before,base_parameters_unchanged=True,trainable_parameters=545,
        sampled_train_row_indices=indices,training_examples_sha256=hashlib.sha256(json.dumps(examples,ensure_ascii=False).encode()).hexdigest(),
        fit_seconds=fit_seconds,history=history,architecture='CopyT5; learned source-copy mixture on frozen own Transformer')
    checkpoint=local/'candidate';checkpoint.mkdir()
    model.save_pretrained(checkpoint);tokenizer.save_pretrained(checkpoint)
    (checkpoint/'step.json').write_text(json.dumps(dict(step=73024,adapter_updates=PLAN['adapter_updates'])))
    (checkpoint/'run_config.json').write_text(json.dumps(config,indent=2))
    torch.save(dict(adapter_updates=PLAN['adapter_updates'],optimizer=optimizer.state_dict()),checkpoint/'copy_optimizer.pt')
    destination=RUN/'candidate'/'step-073024';archive_checkpoint(checkpoint,destination)
    # Reload the exact saved weights so generation verifies the persisted model.
    del model,optimizer
    model=CopyT5.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager').float().cpu().eval()
    assert base_hash(model)==before
    results={}
    for arm,enabled in [('control',False),('candidate',True)]:
        model.copy_enabled=enabled;(RUN/arm).mkdir(exist_ok=True)
        results[arm]={}
        for name,offset,count in [('pilot',256,256),('confirmation',512,512)]:
            results[arm][name]=evaluate(model,tokenizer,heldout[offset:offset+count],RUN/arm/(name+'.json'),offset)
    model.copy_enabled=True
    count,seconds=generate_predictions(destination,destination/'predictions.csv',cache,
        model=model,tokenizer=tokenizer,batch_size=16,num_beams=1,syntax_retry=False,precision='fp32')
    digest=package_jsonl(destination/'predictions.csv',destination/'predictions.jsonl')
    (destination/'results.complete.json').write_text(json.dumps(dict(step=73024,rows=count,
        jsonl_sha256=digest,selection='greedy',bot_status='pending',t4_timing_verified=False),indent=2))
    verification=verify(destination,destination,config['source_commit'])
    (destination/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    gains=[results['candidate'][s]['numeric_correct']-results['control'][s]['numeric_correct'] for s in ['pilot','confirmation']]
    failure_delta=sum(results['candidate'][s]['prediction_failures']-results['control'][s]['prediction_failures'] for s in ['pilot','confirmation'])
    selection=dict(results=results,split_gains=gains,failure_delta=failure_delta,
        eligible=min(gains)>0 and failure_delta<=0,long_run_authorized=False)
    (RUN/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (RUN/'execution.json').write_text(json.dumps(dict(completed=True,gpu_used=False,
        base_parameters_unchanged=True,trainable_parameters=545,elapsed_seconds=time.monotonic()-began,
        generation_seconds=seconds),indent=2)+'\n')
    with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(RUN.rglob('*')):
            if f.is_file() and f.suffix in {'.json','.jsonl','.csv'}:z.write(f,f.relative_to(RUN))
    print('COPY_ADAPTER_COMPLETED',selection['split_gains'],failure_delta,flush=True)


if __name__=='__main__':run()
