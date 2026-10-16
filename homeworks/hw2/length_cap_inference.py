"""CPU greedy regeneration only where baseline sequence limits discarded information."""
import csv
import json
from pathlib import Path
import shutil
import time
import zipfile

from .checkpoint_io import drive_preflight, package_jsonl, sha256
from .short_pilots import ROOT, copy_verified
from .verify_export import verify, QUESTION_ORDER_SHA256

NAME='length-cap-cpu-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'inference'/NAME
PLAN=json.loads((Path(__file__).parent/'experiments'/NAME/'plan.json').read_text())


def affected_rows(question_ids, generated_ids, input_limit, eos_id):
    if len(question_ids)!=len(generated_ids):
        raise ValueError('Question/output row counts differ')
    return [i for i,(q,g) in enumerate(zip(question_ids,generated_ids))
            if len(q)>input_limit or eos_id not in g]


def run():
    import hashlib
    import torch
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast,T5ForConditionalGeneration
    from .data import fetch_parquet
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    torch.set_num_threads(4)
    drive_preflight(ROOT)
    assert not RUN.exists()
    began=time.monotonic()
    source=Path(PLAN['baseline_mount_path'])
    local=Path('/content/length-cap-work');local.mkdir(exist_ok=False)
    model_dir=local/'baseline'/'step-073024';model_dir.mkdir(parents=True)
    manifest=json.loads((source/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==PLAN['model_sha256']
    for name,digest in manifest['files'].items():
        assert Path(name).name==name
        if name!='training_state.pt':
            copy_verified(source/name,model_dir/name)
            assert sha256(model_dir/name)==digest
    for name in ['checkpoint.complete.json','predictions.csv','predictions.jsonl',
                 'predictions.provenance.json','results.complete.json']:
        copy_verified(source/name,model_dir/name)
    baseline=verify(model_dir,model_dir,PLAN['baseline_source_commit'])
    assert baseline['jsonl_sha256']==PLAN['baseline_jsonl_sha256']
    provenance=json.loads((model_dir/'predictions.provenance.json').read_text())
    assert provenance['num_beams']==1 and provenance['syntax_retry_beams']==0
    tokenizer=PreTrainedTokenizerFast.from_pretrained(model_dir,local_files_only=True)
    test=fetch_parquet('data/test/test.parquet',local/'cache')
    questions=pq.read_table(test,columns=['question'])['question'].to_pylist()
    assert hashlib.sha256(json.dumps(questions,ensure_ascii=False).encode()).hexdigest()==QUESTION_ORDER_SHA256
    full=tokenizer(questions,truncation=False,return_token_type_ids=False)['input_ids']
    selected=provenance['selected_token_ids']
    indices=affected_rows(full,selected,PLAN['baseline_input_limit'],tokenizer.eos_token_id)
    assert indices==PLAN['affected_rows']
    model=T5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True).float().eval()
    assert sum(p.numel() for p in model.parameters())==42481152
    assert model.config.eos_token_id==tokenizer.eos_token_id
    started=time.monotonic()
    with torch.inference_mode():
        for offset in range(0,len(indices),PLAN['batch_size']):
            batch_indices=indices[offset:offset+PLAN['batch_size']]
            inputs=tokenizer([questions[i] for i in batch_indices],truncation=True,
                max_length=PLAN['input_limit'],padding=True,return_tensors='pt',return_token_type_ids=False)
            generated=model.generate(**inputs,max_new_tokens=PLAN['output_limit'],
                num_beams=1,do_sample=False,use_cache=True)
            for i,ids in zip(batch_indices,generated.tolist()):selected[i]=ids
            print('GREEDY_ROWS',min(offset+PLAN['batch_size'],len(indices)),len(indices),flush=True)
    seconds=time.monotonic()-started
    out=local/'candidate'/'step-073024';out.mkdir(parents=True)
    codes=tokenizer.batch_decode(selected,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    original=[json.loads(line)['code'] for line in (model_dir/'predictions.jsonl').read_text().splitlines()]
    untouched=set(range(1919))-set(indices)
    assert all(codes[i]==original[i] for i in untouched)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['code']);writer.writeheader();writer.writerows({'code':c} for c in codes)
    digest=package_jsonl(out/'predictions.csv',out/'predictions.jsonl')
    commit=Path('source_commit.txt').read_text().strip()
    provenance.update(source_commit=commit,output_sha256=sha256(out/'predictions.csv'),
        generation_seconds=seconds,gpu=None,precision='torch.float32',batch_size=PLAN['batch_size'],
        max_new_tokens=PLAN['output_limit'],max_input_tokens=PLAN['input_limit'],selected_token_ids=selected,
        baseline_jsonl_sha256=baseline['jsonl_sha256'],baseline_source_commit=PLAN['baseline_source_commit'],
        regenerated_rows=indices,preserved_rows=len(untouched),
        policy='Preserve exact baseline greedy tokens on unaffected rows; regenerate every input-truncated or non-EOS row with larger limits. No answer-based selection or code edits.',
        timing_scope='Incremental CPU generation for affected rows only; excludes prior baseline generation and setup.')
    (out/'predictions.provenance.json').write_text(json.dumps(provenance))
    for name in ['checkpoint.complete.json','step.json','config.json','generation_config.json',
                 'tokenizer.json','tokenizer_config.json','special_tokens_map.json','validation.json','run_config.json']:
        shutil.copyfile(model_dir/name,out/name)
    result=dict(step=73024,rows=1919,jsonl_sha256=digest,generation_seconds=seconds,
        source_commit=commit,selection='greedy',bot_status='pending',t4_timing_verified=False)
    (out/'results.complete.json').write_text(json.dumps(result,indent=2))
    evidence=verify(out,out,commit)
    (out/'verification.json').write_text(json.dumps(evidence,indent=2))
    summary=dict(gpu_used=False,training_performed=False,regenerated_rows=indices,
        changed_programs=sum(a!=b for a,b in zip(codes,original)),
        missing_eos_after=[i for i in indices if tokenizer.eos_token_id not in selected[i]],
        incremental_generation_seconds=seconds,elapsed_seconds=time.monotonic()-began,
        baseline_jsonl_sha256=baseline['jsonl_sha256'],candidate_jsonl_sha256=digest,
        source_commit=commit,t4_timing_verified=False)
    RUN.mkdir(parents=True);destination=RUN/'step-073024';destination.mkdir()
    for f in out.iterdir():copy_verified(f,destination/f.name)
    (RUN/'execution.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(RUN.rglob('*')):
            if f.is_file() and f.suffix in {'.json','.jsonl','.csv'}:z.write(f,f.relative_to(RUN))
    print('LENGTH_CAP_SAVED',json.dumps(summary),flush=True)


if __name__=='__main__':run()
