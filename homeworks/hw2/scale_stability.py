"""Re-score saved held-out generations and measure paired checkpoint changes.

CPU only: no model generation, training, public-test labels, or answer merging.
Every aggregate must reproduce its archived evaluation before it is compared.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

from .numeric_eval import execute_numeric


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, steps, output):
    if output.exists():
        raise ValueError('Preserve the existing audit; choose a new output path')
    if len(steps) < 2 or len(set(steps)) != len(steps):
        raise ValueError('Provide at least two distinct checkpoint steps')
    began = time.monotonic()
    samples, sources, archived = {}, {}, {}
    for step in steps:
        matches = list(root.glob(f'segment-*/step-{step:06d}'))
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous checkpoint {step}')
        folder = matches[0]
        manifest = json.loads((folder/'checkpoint.complete.json').read_text())
        for split in ('pilot', 'confirmation'):
            source = folder/f'{split}.generation.json'
            bundle = json.loads(source.read_text())
            meta = bundle['metadata']
            predictions = folder/f'{split}.predictions.jsonl'
            assert bundle['version'] == 1
            assert digest(predictions) == meta['predictions_sha256']
            assert meta['model_sha256'] == manifest['files']['model.safetensors']
            references = bundle['references']
            assert hashlib.sha256(json.dumps(references, ensure_ascii=False).encode()).hexdigest() == bundle['references_sha256']
            records = [json.loads(line) for line in predictions.read_text().splitlines()]
            assert [r['row'] for r in records] == list(range(meta['offset'], meta['offset']+len(references)))
            samples[step, split] = [(r['code'], ref) for r, ref in zip(records, references)]
            archived[step, split] = json.loads((folder/f'{split}.json').read_text())
            for key in ('model_sha256', 'predictions_sha256', 'questions_sha256', 'evaluation_examples_sha256'):
                assert archived[step, split][key] == meta[key]
            sources[f'{step}/{split}'] = dict(generation_sha256=digest(source),
                predictions_sha256=digest(predictions), score_sha256=digest(folder/f'{split}.json'))
    for split in ('pilot', 'confirmation'):
        base = archived[steps[0], split]
        for step in steps:
            for key in ('questions_sha256', 'evaluation_examples_sha256', 'offset'):
                assert archived[step, split][key] == base[key]
            assert [r for _, r in samples[step, split]] == [r for _, r in samples[steps[0], split]]
    programs = sorted({p for rows in samples.values() for pair in rows for p in pair})
    with ThreadPoolExecutor(max_workers=4) as pool:
        executed = dict(zip(programs, pool.map(execute_numeric, programs)))
    correct, usable, results = {}, {}, {}
    for (step, split), rows in samples.items():
        usable[split] = {i for i, (_, r) in enumerate(rows) if executed[r]['status'] == 'ok'}
        correct[step, split] = {i for i in usable[split]
            if executed[rows[i][0]]['status'] == 'ok' and math.isclose(
                executed[rows[i][0]]['value'], executed[rows[i][1]]['value'], rel_tol=1e-6, abs_tol=1e-6)}
        statuses = Counter(executed[p]['status'] for p, _ in rows)
        result = dict(rows=len(rows), usable_references=len(usable[split]),
            numeric_correct=len(correct[step, split]), prediction_failures=len(rows)-statuses['ok'])
        for key, value in result.items():
            if value != archived[step, split][key]:
                raise ValueError(f'CPU score does not reproduce archived result: {step}/{split}/{key}')
        results[f'{step}/{split}'] = dict(result, prediction_statuses=dict(statuses))
    pairs = []
    for earlier, later in itertools.combinations(steps, 2):
        splits = {}
        for split in ('pilot', 'confirmation'):
            a, b = correct[earlier, split], correct[later, split]
            splits[split] = dict(gained=len(b-a), lost=len(a-b), net_gain=len(b)-len(a),
                both_correct=len(a & b), neither_correct=len(usable[split]-(a | b)))
        pairs.append(dict(earlier=earlier, later=later, splits=splits,
            total_gained=sum(s['gained'] for s in splits.values()),
            total_lost=sum(s['lost'] for s in splits.values())))
    result = dict(steps=steps, device='CPU', workers=4, python_version=platform.python_version(),
        unique_programs_executed=len(programs), elapsed_seconds=time.monotonic()-began,
        scorer_sha256=digest(Path(__file__).with_name('numeric_eval.py')),
        audit_source_sha256=digest(Path(__file__)), source_sha256=sources,
        archived_scores_reproduced=True, results=results, pairs=pairs,
        training_updates=0, public_test_used=False, answers_merged=False,
        limitations=['Agreement with supplied train references is not verified mathematical correctness.',
            'These repeatedly used held-out subsets support exploratory diagnosis, not an unbiased final estimate.',
            'Paired turnover does not establish that ensembling, averaging, or wider models improve accuracy.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('steps', 'elapsed_seconds', 'archived_scores_reproduced', 'pairs')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    audit(**vars(parser.parse_args()))
