"""CPU diagnostic of original programs; never creates a training target file."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import pyarrow.parquet as pq

from .numeric_eval import execute_numeric

SEED = 20261016


def sample_key(question, seed=SEED):
    return hashlib.blake2b(question.encode('utf-8'), digest_size=16,
                           key=str(seed).encode('ascii')).digest()


def classify(results):
    values = [r['value'] for r in results if r['status'] == 'ok']
    conflicts = sum(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
                    for a, b in itertools.combinations(values, 2))
    if conflicts:
        category = 'numeric_conflict'
    elif len(values) < 2:
        category = 'insufficient_numeric'
    elif len(values) == len(results):
        category = 'all_executable_agree'
    else:
        category = 'partial_agreement'
    return dict(category=category, executable_programs=len(values),
                compared_pairs=len(values)*(len(values)-1)//2,
                conflicting_pairs=conflicts, statuses=dict(Counter(r['status'] for r in results)))


def audit(cache, output, *, max_shards=60, modulus=1024, max_groups=512,
          max_programs=8, heldout_rows=1024):
    if not 1 <= max_shards <= 60 or modulus < 1 or max_groups < 1 or max_programs < 2:
        raise ValueError('invalid audit bounds')
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    heldout = set(pq.read_table(cache/'part-00000.parquet', columns=['question'])
                  .slice(0, heldout_rows)['question'].to_pylist())
    groups, shards = {}, []
    for shard in range(max_shards):
        path = cache/f'part-{shard:05d}.parquet'
        table = pq.read_table(path, columns=['question', 'code'])
        questions = table['question'].to_pylist()
        eligible = sampled = 0
        for i, question in enumerate(questions):
            if question in heldout:
                continue
            eligible += 1
            if int.from_bytes(sample_key(question), 'big') % modulus:
                continue
            sampled += 1
            code = table['code'][i].as_py()
            group = groups.setdefault(question, {'rows': 0, 'programs': {}})
            group['rows'] += 1
            # Group by exact strings; digests are only output identifiers.
            group['programs'].setdefault(code, []).append([shard, i])
        shards.append(dict(shard=shard, rows=len(table), eligible=eligible, sampled=sampled,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (output/'progress.json').write_text(json.dumps(dict(stage='scan', shards_completed=shard+1,
            sampled_question_groups=len(groups), elapsed_seconds=time.monotonic()-began)))
        print('SCAN', shard+1, 'sampled_groups', len(groups), flush=True)
    repeated = [(q, g) for q, g in groups.items() if g['rows'] > 1]
    distinct = sorted(((q, g) for q, g in repeated if len(g['programs']) > 1),
                      key=lambda item: sample_key(item[0]))
    selected = distinct[:max_groups]
    evidence = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (question, group) in enumerate(selected):
            codes = sorted(group['programs'], key=lambda c: hashlib.sha256(c.encode()).digest())[:max_programs]
            results = list(pool.map(execute_numeric, codes))
            record = dict(question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                          original_rows=group['rows'], distinct_programs=len(group['programs']),
                          tested_programs=len(codes), program_cap_applied=len(codes)<len(group['programs']),
                          **classify(results),
                          programs=[dict(sha256=hashlib.sha256(code.encode()).hexdigest(),
                                         source_rows=group['programs'][code], execution_status=result['status'])
                                    for code, result in zip(codes, results)])
            evidence.append(record)
            (output/'progress.json').write_text(json.dumps(dict(stage='execute', groups_completed=index+1,
                groups_total=len(selected), elapsed_seconds=time.monotonic()-began)))
            if (index+1) % 50 == 0:
                print('EXECUTE', index+1, '/', len(selected), flush=True)
    payload = ''.join(json.dumps(row)+'\n' for row in evidence).encode()
    (output/'groups.jsonl').write_bytes(payload)
    result = dict(purpose='CPU audit only; original programs unchanged, no training targets generated',
        seed=SEED, question_sampling_modulus=modulus, max_groups=max_groups, max_programs=max_programs,
        shards=shards, eligible_rows=sum(s['eligible'] for s in shards),
        sampled_rows=sum(s['sampled'] for s in shards), sampled_question_groups=len(groups),
        repeated_question_groups=len(repeated), identical_program_groups=len(repeated)-len(distinct),
        different_program_groups=len(distinct), tested_groups=len(evidence),
        tested_programs=sum(e['tested_programs'] for e in evidence),
        capped_groups=sum(e['program_cap_applied'] for e in evidence),
        categories=dict(Counter(e['category'] for e in evidence)),
        execution_statuses=dict(sum((Counter(e['statuses']) for e in evidence), Counter())),
        groups_sha256=hashlib.sha256(payload).hexdigest(), elapsed_seconds=time.monotonic()-began,
        exclusion='Every exact copy of the first 1024 held-out question texts across all scanned shards',
        selection='Keyed question hash modulo sampling, then smallest keyed hashes among distinct-program repeated groups; independent of test/public examples',
        limitations='Exact strings only; sampled groups are not row-weighted. Numeric disagreement does not identify truth. Agreement does not establish correctness. Unsupported execution is unresolved, not a bad target. A program cap can miss conflicts. Existing restricted diagnostic executor is not the course bot sandbox.')
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps({k:v for k,v in result.items() if k!='shards'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('CPU only: do not allocate a GPU for this audit')
    audit(args.cache, args.output)
