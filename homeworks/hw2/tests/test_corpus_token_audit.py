"""Synthetic audit fixtures only; never used as training data."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from homeworks.hw2.corpus_token_audit import audit, count_texts, histogram_summary, sha256


TOKENIZER=Path('homeworks/hw2/tokenizer/tokenizer.json')


def test_histogram_quantiles_match_expanded_lengths():
    values=np.array([1,1,2,3,3,3,100])
    result=histogram_summary(dict(zip(*np.unique(values,return_counts=True))))
    assert result['total_tokens']==values.sum()
    for q,value in result['quantiles'].items():
        assert value==np.quantile(values,float(q),method='higher')


def test_full_audit_exclusions_caps_and_verified_resume(tmp_path):
    cache=tmp_path/'cache';index=tmp_path/'index';out=tmp_path/'output'
    cache.mkdir();index.mkdir()
    rows=[{'question':'a '*n,'code':'print(1)\n'*n} for n in [1,2,80,100,200]]
    data=cache/'part-00000.parquet';pq.write_table(pa.Table.from_pylist(rows),data)
    ids=np.array([0,2,4],dtype=np.int64);np.save(index/'00000.npy',ids)
    record=dict(shard=0,filename=data.name,rows=5,eligible=3,excluded=2,
                parquet_sha256=sha256(data),index_sha256=sha256(index/'00000.npy'))
    (index/'manifest.json').write_text(json.dumps(dict(version=1,shards=[record])))
    result=audit(cache,index,TOKENIZER,out)
    tok=Tokenizer.from_file(str(TOKENIZER));tok.no_padding();tok.no_truncation()
    q,qs=count_texts(tok,[rows[i]['question'] for i in ids]);c,cs=count_texts(tok,[rows[i]['code'] for i in ids])
    assert result['eligible_rows']==3
    assert result['results']['question']['total_tokens']==q.sum()
    assert result['results']['code']['total_tokens']==c.sum()
    assert result['caps']['128:192']['either_cut']==sum(a>128 or b>192 for a,b in zip(q,c))
    assert result['caps']['128:192']['target_tokens_removed']==sum(max(int(n)-192,0) for n in c)
    assert result['special_tokens']=={'question':qs,'code':cs}
    resumed=audit(cache,index,TOKENIZER,out)
    assert resumed['results']==result['results'] and resumed['caps']==result['caps']
    with open(out/'shard-00000.npz','ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='Changed audit input or artifact'):
        audit(cache,index,TOKENIZER,out)


def test_non_string_is_not_silently_dropped():
    tok=Tokenizer.from_file(str(TOKENIZER));tok.no_padding();tok.no_truncation()
    with pytest.raises(ValueError,match='Null/non-string'):
        count_texts(tok,['text',None])
