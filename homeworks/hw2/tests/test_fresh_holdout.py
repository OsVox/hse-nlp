import numpy as np
import pytest
from homeworks.hw2.fresh_holdout import choose, exclude_indices


def test_reserves_question_groups_without_consulting_programs():
    rows=[{'question':q,'code':str(i)} for i,q in enumerate(['old','a','b','a','c'])]
    picked=choose(rows,{'old'},count=3)
    assert {r['question'] for _,r in picked}=={'a','b','c'}
    assert next(i for i,r in picked if r['question']=='a')==1
    changed=[dict(r,code='different') for r in rows]
    assert [i for i,_ in picked]==[i for i,_ in choose(changed,{'old'},count=3)]
    with pytest.raises(ValueError):choose(rows,{'old'},count=4)


def test_removes_all_reserved_copies_without_reintroducing_old_exclusions():
    original=np.asarray([1,2,3,5],dtype=np.int64)
    filtered=exclude_indices(['old','a','b','a','c','d'],original,{'a','c'})
    assert filtered.tolist()==[2,5]
    assert original.tolist()==[1,2,3,5]


def test_stratified_selection_balances_sources_excludes_duplicate_groups_and_ignores_answers():
    from homeworks.hw2.fresh_holdout import choose_stratified
    shards=[(i,[dict(question=q,code=str(j)) for j,q in enumerate(['forbidden','shared']+[f'{i}-{j}' for j in range(20)])]) for i in range(1,4)]
    selected=choose_stratified(shards,{'forbidden'},count=8)
    assert [sum(s==i for s,_,_ in selected) for i in range(1,4)]==[3,3,2]
    questions=[r['question'] for _,_,r in selected]
    assert len(set(questions))==8 and 'forbidden' not in questions
    changed=[(s,[dict(r,code='changed') for r in rows]) for s,rows in shards]
    assert [(s,i) for s,i,_ in selected]==[(s,i) for s,i,_ in choose_stratified(changed,{'forbidden'},count=8)]
