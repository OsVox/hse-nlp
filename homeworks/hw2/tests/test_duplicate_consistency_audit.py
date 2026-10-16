import json

import pyarrow as pa
import pyarrow.parquet as pq

from homeworks.hw2.duplicate_consistency_audit import audit, classify, sample_key


def test_classification_distinguishes_conflict_from_unsupported_execution():
    ok=lambda v: {'status':'ok','value':v}
    assert classify([ok(9),ok(9+1e-8)])['category']=='all_executable_agree'
    assert classify([ok(9),ok(4.5)])['category']=='numeric_conflict'
    assert classify([ok(9),{'status':'ValueError'}])['category']=='insufficient_numeric'
    assert classify([ok(9),ok(9),{'status':'timeout'}])['category']=='partial_agreement'
    assert sample_key('same')==sample_key('same')!=sample_key('different')


def test_original_group_scan_excludes_heldout_copies_and_preserves_sources(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    def code(n):return f'def solve():\n return {n}'
    rows0=[{'question':'reserved','code':code(0)},
           {'question':'conflict','code':code(2)},
           {'question':'agree','code':code(9)},
           {'question':'identical','code':code(3)},
           {'question':'unique','code':code(4)}]
    rows1=[{'question':'reserved','code':code(100)},
           {'question':'conflict','code':code(3)},
           {'question':'agree','code':code('18/2')},
           {'question':'identical','code':code(3)}]
    originals={}
    for i,rows in enumerate([rows0,rows1]):
        p=cache/f'part-{i:05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p);originals[p]=p.read_bytes()
    result=audit(cache,tmp_path/'out',max_shards=2,modulus=1,heldout_rows=1)
    assert result['eligible_rows']==7
    assert result['repeated_question_groups']==3 and result['identical_program_groups']==1
    assert result['categories']=={'numeric_conflict':1,'all_executable_agree':1}
    assert all(p.read_bytes()==data for p,data in originals.items())
    records=[json.loads(line) for line in (tmp_path/'out/groups.jsonl').read_text().splitlines()]
    assert all('value' not in p and 'code' not in p for r in records for p in r['programs'])
    assert sorted(pos for r in records for p in r['programs'] for pos in p['source_rows'])==[[0,1],[0,2],[1,1],[1,2]]
