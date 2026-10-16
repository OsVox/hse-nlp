"""Reserve deterministic original question groups for future scratch-model tests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_io import sha256
from .full_coverage import digest, load_index


def choose(rows, forbidden, count=2048, seed=20261017):
    candidates = {}
    for i, row in enumerate(rows):
        question = row['question']
        if question not in forbidden and question not in candidates:
            candidates[question] = (i,row)
    if len(candidates) < count:
        raise ValueError('Not enough distinct unseen question texts')
    ordered = sorted(candidates, key=lambda q:(hashlib.sha256(f'{seed}\n{q}'.encode()).hexdigest(),q))
    return [candidates[q] for q in ordered[:count]]


def exclude_indices(questions, eligible, reserved):
    return np.asarray([int(i) for i in eligible if questions[int(i)] not in reserved], dtype=np.int64)


def choose_stratified(shards, forbidden, count=2048, seed=20261017):
    """Balance source shards, exclude duplicate groups, then mix split positions."""
    shards = list(shards)
    if not shards or count < len(shards):
        raise ValueError('Need at least one question per source shard')
    blocked = set(forbidden)
    selected = []
    for position, (shard, rows) in enumerate(shards):
        quota = count // len(shards) + (position < count % len(shards))
        picks = choose(rows() if callable(rows) else rows, blocked, quota, seed)
        selected.extend((shard, i, row) for i, row in picks)
        blocked.update(row['question'] for _, row in picks)
    return sorted(selected, key=lambda x: hashlib.sha256(
        f'split:{seed}\n{x[2]["question"]}'.encode()).hexdigest())


def prepare(cache, old_index, output, evidence, stratified=False):
    assert not output.exists(), 'Preserve previous splits'
    manifest, paths, indices = load_index(cache,old_index)
    first = pq.read_table(paths[0],columns=['question'])['question'].to_pylist()
    # The fixed tokenizer was fitted only on shard zero. Exclude its question
    # column; overlap with text embedded in original programs is not audited.
    if stratified:
        selected = choose_stratified([(i, lambda path=path: pq.read_table(
            path, columns=['question','code']).to_pylist()) for i,path in enumerate(paths) if i],set(first))
    else:
        last = pq.read_table(paths[-1],columns=['question','code']).to_pylist()
        selected = [(len(paths)-1,i,row) for i,row in choose(last,set(first))]
    reserved = {row['question'] for _,_,row in selected}
    assert len(reserved)==2048 and not reserved.intersection(first)
    output.mkdir(parents=True)
    (output/'index').mkdir()
    questions = output/'questions.jsonl'
    with questions.open('w') as f:
        for position,(shard,i,row) in enumerate(selected):
            f.write(json.dumps(dict(position=position,shard=shard,row=i,
                question=row['question'],code=row['code']),ensure_ascii=False)+'\n')
    records=[]
    for path, eligible, old in zip(paths,indices,manifest['shards']):
        texts=pq.read_table(path,columns=['question'])['question'].to_pylist()
        filtered=exclude_indices(texts,eligible,reserved)
        assert not any(texts[int(i)] in reserved for i in filtered)
        index_path=output/'index'/f"{old['shard']:05d}.npy"
        np.save(index_path,filtered,allow_pickle=False)
        records.append({**old,'eligible':len(filtered),'excluded':len(texts)-len(filtered),
                        'index_sha256':sha256(index_path)})
    combined = set(first[:1024]) | reserved
    updated=dict(version=1,shards=records,heldout_question_sha256=digest(sorted(combined)),
                 additional_heldout=dict(count=2048,questions_sha256=sha256(questions),
                    selection_seed=20261017,source_shards=sorted({s for s,_,_ in selected}),
                    stratified=stratified,
                    forbidden_question_source='Every question text in tokenizer-fitting shard zero'))
    (output/'index/manifest.json').write_text(json.dumps(updated,indent=2)+'\n')
    before=sum(r['eligible'] for r in manifest['shards'])
    after=sum(r['eligible'] for r in records)
    result=dict(cpu_verified=True,question_groups=2048,source_shards=sorted({s for s,_,_ in selected}),
                source_shard_counts={str(s):sum(x[0]==s for x in selected) for s in sorted({x[0] for x in selected})},
                stratified=stratified,
                old_eligible_rows=before,new_eligible_rows=after,additional_excluded_rows=before-after,
                questions_sha256=sha256(questions),index_manifest_sha256=sha256(output/'index/manifest.json'),
                pilot_positions=[0,1024],confirmation_positions=[1024,2048],
                all_60_shards_checked=len(records)==60,tokenizer_question_overlap=0,
                original_programs_unchanged=True,selection_used_answers=False,
                validity='Clean holdout only for future scratch models trained with this derived exclusion index. Historical checkpoints may have trained on these questions; do not claim they are unseen by those models.',
                reference_limit='Supplied programs can disagree or be unsupported by the numeric runner. Report usable-reference denominators; do not rewrite or select references based on results.')
    evidence.parent.mkdir(parents=True,exist_ok=True)
    evidence.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','old-index','output','evidence'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stratified',action='store_true')
    prepare(**vars(p.parse_args()))
