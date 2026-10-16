"""Audit paired held-out changes using saved generations; no model or GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path

from .numeric_eval import execute_numeric


def audit(root, output):
    samples = {}
    sources = {}
    for step in (73728, 74752, 75776, 76800):
        folder = root / 'complete' / f'step-{step:06d}'
        if step == 76800:
            folder = root / 'complete'
        rows = {}
        for split in ('pilot', 'confirmation'):
            source = folder / f'{split}.generation.json'
            bundle = json.loads(source.read_text())
            meta = bundle['metadata']
            predictions = folder / f'{split}.predictions.jsonl'
            assert hashlib.sha256(predictions.read_bytes()).hexdigest() == meta['predictions_sha256']
            references = bundle['references']
            assert hashlib.sha256(json.dumps(references, ensure_ascii=False).encode()).hexdigest() == bundle['references_sha256']
            records = [json.loads(line) for line in predictions.read_text().splitlines()]
            assert [r['row'] for r in records] == list(range(meta['offset'], meta['offset'] + len(references)))
            for record, reference in zip(records, references):
                assert record['row'] not in rows
                rows[record['row']] = (record['code'], reference)
            sources[f'{step}/{split}'] = hashlib.sha256(source.read_bytes()).hexdigest()
        samples[step] = rows
    row_ids = set(samples[73728])
    assert all(set(rows) == row_ids for rows in samples.values())
    assert all(len({rows[i][1] for rows in samples.values()}) == 1 for i in row_ids)
    programs = sorted({code for rows in samples.values() for pair in rows.values() for code in pair})
    with ThreadPoolExecutor(max_workers=4) as pool:
        executed = dict(zip(programs, pool.map(execute_numeric, programs)))
    correct = {}
    usable = {i for i in row_ids if executed[samples[73728][i][1]]['status'] == 'ok'}
    failures = {}
    for step, rows in samples.items():
        correct[step] = set()
        failures[step] = sum(executed[p]['status'] != 'ok' for p, _ in rows.values())
        for i in usable:
            p, r = map(executed.get, rows[i])
            if p['status'] == 'ok' and math.isclose(p['value'], r['value'], rel_tol=1e-6, abs_tol=1e-6):
                correct[step].add(i)
    result = dict(device='CPU', training_updates=0, public_test_used=False,
                  rows=len(row_ids), usable_references=len(usable),
                  correct={k:len(v) for k,v in correct.items()}, failures=failures,
                  source_sha256=sources, pairs=[])
    for a, b in itertools.combinations(samples, 2):
        result['pairs'].append(dict(earlier=a, later=b,
            gained=len(correct[b]-correct[a]), lost=len(correct[a]-correct[b]),
            both_correct=len(correct[a]&correct[b]),
            neither_correct=len(usable-(correct[a]|correct[b]))))
    result['interpretation'] = ('Paired disagreement diagnoses instability only. It does not demonstrate that weight averaging improves accuracy; no answers are combined or repaired.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    audit(**vars(parser.parse_args()))
