"""Inspect target truncation on a reproducible sample; never execute programs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .corpus_token_audit import sha256


def syntax(code):
    try:
        tree=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError):
        return {'parses':False,'has_return':None}
    return {'parses':True,'has_return':any(isinstance(n,ast.Return) for n in ast.walk(tree))}


def audit(cache,index,lengths,tokenizer,output):
    census=json.loads((lengths/'results.json').read_text())
    assert census['shards']==60 and census['eligible_rows']==11836444
    assert sha256(tokenizer)==census['identity']['tokenizer_sha256']
    assert sha256(index/'manifest.json')==census['identity']['index_manifest_sha256']
    tok=Tokenizer.from_file(str(tokenizer));tok.no_padding();tok.enable_truncation(192)
    eos=tok.token_to_id('</s>');records=[]
    for shard in range(60):
        evidence=json.loads((lengths/f'shard-{shard:05d}.json').read_text())
        path=cache/evidence['record']['filename'];ids_path=index/f'{shard:05d}.npy'
        assert sha256(path)==evidence['record']['parquet_sha256']
        assert sha256(ids_path)==evidence['record']['index_sha256']
        array=lengths/f'shard-{shard:05d}.npz';assert sha256(array)==evidence['length_file_sha256']
        with np.load(array,allow_pickle=False) as a:
            long=np.flatnonzero(a['code']>192)
            chosen=np.random.default_rng(np.random.SeedSequence([42,shard])).choice(long,size=min(32,len(long)),replace=False)
            token_lengths=a['code'][chosen]
        ids=np.load(ids_path,allow_pickle=False)[chosen]
        codes=pq.read_table(path,columns=['code']).take(ids)['code'].to_pylist()
        for row,n,code,encoding in zip(ids,token_lengths,codes,tok.encode_batch(codes)):
            assert len(encoding.ids)==192 and encoding.ids[-1]==eos
            prefix=tok.decode(encoding.ids,skip_special_tokens=True)
            records.append(dict(shard=shard,row=int(row),original_tokens=int(n),
                original_sha256=hashlib.sha256(code.encode()).hexdigest(),
                truncated_decoded_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                original=syntax(code),truncated=syntax(prefix)))
    valid=[r for r in records if r['original']['parses']]
    broken=sum(not r['truncated']['parses'] for r in valid)
    missing=sum(r['truncated']['parses'] and r['original']['has_return'] and not r['truncated']['has_return'] for r in valid)
    result=dict(seed=42,sampled_per_shard_up_to=32,sample_rows=len(records),
        original_parseable=len(valid),original_unparseable=len(records)-len(valid),
        became_unparseable=broken,parseable_but_lost_all_returns=missing,
        still_parseable_with_return=sum(r['truncated']['parses'] and r['truncated']['has_return'] for r in valid),
        scope='Equal-sized per-shard sample of original eligible targets longer than 192 tokens; not a population-weighted estimate.',
        limitations=['Parsing does not establish execution or numeric correctness.','Keeping a return node does not prove unchanged semantics.','No code is executed, corrected or used as new training data.'],
        census_identity=census['identity'],audit_source_sha256=sha256(__file__),records=records)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['cache','index','lengths','tokenizer','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();audit(a.cache,a.index,a.lengths,a.tokenizer,a.output)
