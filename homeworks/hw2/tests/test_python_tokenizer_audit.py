import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from homeworks.hw2.python_tokenizer_audit import build_tokenizer,collect_rows,metrics


@pytest.mark.parametrize('python_boundaries',[False,True])
def test_code_tokenizer_preserves_text_after_serialization(tmp_path,python_boundaries):
    rows=[{'question':'Find 2.50 plus 12? Café α',
           'code':'def solve():\n\tvalue = 2.50 + 12\n\treturn value  # café α\n'},
          {'question':'A ratio of 3:4',
           'code':'def solve():\r\n    return (3 ** 2) // 4\r\n'}]
    tokenizer=build_tokenizer(rows,code_repeats=3,vocab_size=512,python_boundaries=python_boundaries)
    path=tmp_path/'tokenizer.json';tokenizer.save(str(path))
    loaded=Tokenizer.from_file(str(path));r=metrics(loaded,rows)
    for col in ['question','code']:
        assert r[col]['roundtrip_failures']==r[col]['unknown_tokens']==0
    assert [loaded.token_to_id(s) for s in ['<pad>','</s>','<unk>']]==[0,1,2]


def test_sampling_has_disjoint_question_texts(tmp_path):
    rows=[{'question':f'q{i//2}','code':f'def solve():\n return {i}'} for i in range(40)]
    for shard in range(2):pq.write_table(pa.Table.from_pylist(rows),tmp_path/f'part-{shard:05d}.parquet')
    fit,evaluation,proof=collect_rows(tmp_path,max_shards=2,fit_per_shard=5,eval_per_shard=10,heldout_rows=2)
    assert not ({r['question'] for r in fit}&{r['question'] for r in evaluation})
    assert all(r['question']!='q0' for r in fit+evaluation)
    assert proof['fit_rows']==10 and proof['evaluation_rows']>0
