"""CPU-only audit of information discarded by fixed sequence limits."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
from .data import fetch_parquet
from .verify_export import QUESTION_ORDER_SHA256


def describe(lengths, cap):
    ordered=sorted(lengths)
    return {'rows':len(lengths),'cap':cap,'over_cap':sum(x>cap for x in lengths),
            'max':max(lengths),'mean':sum(lengths)/len(lengths),
            'p95':ordered[int(.95*(len(ordered)-1))],
            'p99':ordered[int(.99*(len(ordered)-1))]}


def run(checkpoint, cache, output):
    began=time.monotonic()
    tokenizer=PreTrainedTokenizerFast.from_pretrained(checkpoint,local_files_only=True)
    test=fetch_parquet('data/test/test.parquet',cache)
    questions=pq.read_table(test,columns=['question'])['question'].to_pylist()
    assert len(questions)==1919
    assert hashlib.sha256(json.dumps(questions,ensure_ascii=False).encode()).hexdigest()==QUESTION_ORDER_SHA256
    manifest=json.loads((checkpoint/'checkpoint.complete.json').read_text())
    assert hashlib.sha256((checkpoint/'tokenizer.json').read_bytes()).hexdigest()==manifest['files']['tokenizer.json']
    encoded=tokenizer(questions,truncation=False,return_token_type_ids=False)['input_ids']
    train=fetch_parquet('data/train/part-00000.parquet',cache)
    rows=pq.read_table(train,columns=['question','code']).slice(0,1024).to_pylist()
    qids=tokenizer([r['question'] for r in rows],truncation=False)['input_ids']
    cids=tokenizer([r['code'] for r in rows],truncation=False)['input_ids']
    provenance=json.loads((checkpoint/'predictions.provenance.json').read_text())
    cap=provenance['max_new_tokens'];tokens=provenance['selected_token_ids']
    missing_eos=[i for i,ids in enumerate(tokens) if tokenizer.eos_token_id not in ids]
    result={'gpu_used':False,'training_performed':False,'test_input':describe([len(x) for x in encoded],128),
        'test_rows_over_input_cap':[i for i,x in enumerate(encoded) if len(x)>128],
        'heldout_input':describe([len(x) for x in qids],128),
        'heldout_target':describe([len(x) for x in cids],192),
        'generated_output_cap':cap,'generated_missing_eos_rows':missing_eos,
        'test_questions_sha256':QUESTION_ORDER_SHA256,
        'test_parquet_sha256':hashlib.sha256(test.read_bytes()).hexdigest(),
        'heldout_shard_sha256':hashlib.sha256(train.read_bytes()).hexdigest(),
        'tokenizer_sha256':manifest['files']['tokenizer.json'],
        'model_sha256':provenance['model_sha256'],'checkpoint':str(checkpoint),
        'scope':'Test questions only, existing generated token IDs, and first 1024 original training rows. No public test labels or new targets.',
        'elapsed_seconds':time.monotonic()-began}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('homeworks/hw2/cache'))
    p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
