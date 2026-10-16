import json
import pytest
from homeworks.hw2.numeric_eval import load_examples


def test_explicit_examples_preserve_order_and_programs(tmp_path):
    rows=[dict(position=i,question=q,code='def solve():\n return 1\n') for i,q in enumerate(['b','a'])]
    p=tmp_path/'examples.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert load_examples(tmp_path,p)==rows


def test_invalid_positions_and_duplicate_groups_rejected(tmp_path):
    p=tmp_path/'examples.jsonl'
    for rows in ([dict(position=1,question='a',code='x')],
                 [dict(position=i,question='a',code='x') for i in range(2)]):
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError):load_examples(tmp_path,p)
