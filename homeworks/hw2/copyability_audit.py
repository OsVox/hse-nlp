"""Measure source-token copying opportunities on reserved training rows, CPU only.

No parameters are fitted and no alternative predictions are submitted. Exact
token availability is necessary for a token-copy head, but does not prove that
the model can choose the right token or solve the complete program correctly.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from .checkpoint_io import sha256
from .data import validation_examples
from .train import encode_batch


def count_group(mask, correct, available, pointer_correct):
    errors = mask & ~correct
    return dict(tokens=int(mask.sum()), correct=int((mask & correct).sum()),
                source_copyable=int((mask & available).sum()),
                errors=int(errors.sum()), copyable_errors=int((errors & available).sum()),
                pointer_correct=int((mask & pointer_correct).sum()),
                pointer_correct_on_errors=int((errors & pointer_correct).sum()))


def run(checkpoint, cache, output, source_commit):
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    assert not output.exists(), 'Use a fresh audit output'
    expected='cad272fa2644c3876a577affc6f8559be9d5efce4e0f5c66f8f3ea6b7fb84cb6'
    manifest=json.loads((checkpoint/'checkpoint.complete.json').read_text())
    for name in ['model.safetensors','config.json','tokenizer.json']:
        assert sha256(checkpoint/name)==manifest['files'][name]
    assert sha256(checkpoint/'model.safetensors')==expected
    torch.set_num_threads(4)
    began=time.monotonic()
    tokenizer=PreTrainedTokenizerFast.from_pretrained(checkpoint,local_files_only=True)
    model=T5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,
        attn_implementation='eager').float().cpu().eval()
    examples=validation_examples(cache)[:256]
    kinds=torch.zeros(len(tokenizer),dtype=torch.long)
    special=set(tokenizer.all_special_ids)
    for i in range(len(tokenizer)):
        if i in special:continue
        piece=tokenizer.decode([i],clean_up_tokenization_spaces=False).strip()
        if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',piece):kinds[i]=1
        elif re.fullmatch(r'[+*/%\-]+',piece):kinds[i]=2
    totals={k:Counter() for k in ['number_piece','operator_piece','other','all']}
    heads={};row_counts=Counter();available_counts=Counter()
    with torch.inference_mode():
        for start in range(0,256,8):
            batch=encode_batch(examples[start:start+8],tokenizer,torch.device('cpu'))
            labels=batch['labels'];valid=labels.ne(-100);label_kind=kinds[labels.clamp_min(0)]
            source_valid=batch['attention_mask'].bool()
            for i in special:source_valid &= batch['input_ids'].ne(i)
            matches=labels[:,:,None].eq(batch['input_ids'][:,None,:]) & source_valid[:,None,:]
            available=matches.any(-1)&valid
            out=model(**batch,use_cache=False,output_attentions=True)
            correct=out.logits.argmax(-1).eq(labels)&valid
            # Inspect all heads without selecting or applying a new decoder.
            pointer_sum=torch.zeros_like(out.logits,dtype=torch.float32)
            for layer,attention in enumerate(out.cross_attentions):
                for head in range(attention.shape[1]):
                    a=attention[:,head].float()*source_valid[:,None,:]
                    a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
                    dist=torch.zeros_like(out.logits,dtype=torch.float32)
                    dist.scatter_add_(2,batch['input_ids'][:,None,:].expand_as(a),a)
                    ptr_correct=dist.argmax(-1).eq(labels)&valid
                    key=f'layer_{layer}_head_{head}'
                    stats=heads.setdefault(key,Counter())
                    numeric=valid&label_kind.eq(1)
                    stats.update(count_group(numeric,correct,available,ptr_correct))
                    pointer_sum.add_(dist)
            mean_pointer=pointer_sum.argmax(-1).eq(labels)&valid
            masks={'number_piece':valid&label_kind.eq(1),'operator_piece':valid&label_kind.eq(2),
                   'other':valid&label_kind.eq(0),'all':valid}
            for name,mask in masks.items():
                totals[name].update(count_group(mask,correct,available,mean_pointer))
                row_counts[name]+=int((mask&~correct&available).any(-1).sum())
                available_counts[name]+=int((mask&available).any(-1).sum())
            print('COPY_AUDIT_ROWS',start+len(labels),256,flush=True)
    result=dict(source_commit=source_commit,model_sha256=expected,
        shard_sha256=sha256(cache/'part-00000.parquet'),tokenizer_sha256=sha256(checkpoint/'tokenizer.json'),
        examples_sha256=hashlib.sha256(json.dumps(examples,ensure_ascii=False).encode()).hexdigest(),
        rows=256,scope='Existing reserved original train shard 0 rows 0:256; no public examples',
        device='cpu',training=False,seconds=time.monotonic()-began,
        groups={k:dict(v) for k,v in totals.items()},
        rows_with_copyable_error=dict(row_counts),rows_with_copyable_target=dict(available_counts),
        number_piece_by_attention_head={k:dict(v) for k,v in heads.items()},
        limitations=['Teacher-forced supplied-reference agreement is not program correctness.',
            'Number/operator pieces are lexical tokens and can occur in comments or strings.',
            'Exact token-ID matching excludes different whitespace/BPE forms of the same numeral.',
            'Attention-head statistics are descriptive; no head was selected for inference.',
            'Copyable errors are an oracle opportunity count, not an attainable improvement estimate.'])
    output.mkdir(parents=True);(output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='number_piece_by_attention_head'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['checkpoint','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args();run(a.checkpoint,a.cache,a.output,a.source_commit)
