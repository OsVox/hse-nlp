"""Index original question frequencies for optional inverse-frequency sampling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def keep_row(frequency, power, rng):
    if not 0 <= power <= 1 or frequency < 0:
        raise ValueError('invalid frequency or sampling power')
    if frequency == 0:
        return False
    return power == 0 or frequency == 1 or rng.random() < frequency ** -power


def build(cache, output, *, shards=60, heldout_rows=1024, sample_per_shard=4096):
    """Two CPU passes; outputs counts only, never question/program targets."""
    if not 1 <= shards <= 60 or heldout_rows < 1 or sample_per_shard < 1:
        raise ValueError('invalid bounds')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    heldout = set(pq.read_table(cache/'part-00000.parquet', columns=['question'])
                  .slice(0, heldout_rows)['question'].to_pylist())
    counts, records = Counter(), []
    for shard in range(shards):
        path = cache/f'part-{shard:05d}.parquet'
        questions = pq.read_table(path, columns=['question'])['question'].to_pylist()
        counts.update(q for q in questions if q not in heldout)
        records.append(dict(file=path.name, rows=len(questions), sha256=digest(path)))
        (output/'progress.json').write_text(json.dumps(dict(stage='count', shards=shard+1)))
        print('COUNT', shard+1, flush=True)
    if not counts:
        raise ValueError('no eligible training questions')
    assert max(counts.values()) <= np.iinfo(np.uint32).max
    candidate_counts = Counter()
    accepted = {str(a): Counter() for a in (0, .5, 1)}
    sample_rng = random.Random(42)
    sample_rnds = {a: random.Random(42 + 2026101603) for a in accepted}
    repeated_mass = {str(a): 0.0 for a in (0, .5, 1)}
    total_mass = dict.fromkeys(repeated_mass, 0.0)
    for shard, record in enumerate(records):
        path = cache/record['file']
        assert digest(path) == record['sha256'], 'source changed between passes'
        questions = pq.read_table(path, columns=['question'])['question'].to_pylist()
        frequencies = np.fromiter((0 if q in heldout else counts[q] for q in questions),
                                  dtype=np.uint32, count=len(questions))
        index = output/(path.stem+'.npy')
        np.save(index, frequencies, allow_pickle=False)
        record.update(index=index.name, index_sha256=digest(index),
                      excluded=int((frequencies == 0).sum()))
        first = heldout_rows if shard == 0 else 0
        # Replicate the candidate drawing order of mixed training seed 42.
        indices = sample_rng.sample(range(first, len(questions)),
                                    min(sample_per_shard, len(questions)-first))
        for i in indices:
            q, f = questions[i], int(frequencies[i])
            if f == 0:
                continue
            candidate_counts[q] += 1
            for power, selected in accepted.items():
                p = f ** -float(power)
                total_mass[power] += p
                repeated_mass[power] += p * (f > 1)
                if keep_row(f, float(power), sample_rnds[power]):
                    selected[q] += 1
        (output/'progress.json').write_text(json.dumps(dict(stage='index', shards=shard+1)))
        print('INDEX', shard+1, flush=True)
    manifest = dict(version=1, shards=records, heldout_rows=heldout_rows,
                    heldout_questions_sha256=hashlib.sha256(
                        json.dumps(sorted(heldout), ensure_ascii=False).encode()).hexdigest(),
                    eligible_rows=sum(counts.values()), unique_questions=len(counts),
                    grouping='Exact question strings; no hash grouping collisions',
                    source_targets_changed=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = dict(eligible_rows=sum(counts.values()), unique_questions=len(counts),
        candidate_rows=sum(candidate_counts.values()), candidate_unique_questions=len(candidate_counts),
        sample_seed=42, sample_per_shard=sample_per_shard,
        arms={power: dict(accepted_rows=sum(c.values()), unique_questions=len(c),
            repeated_question_rows=sum(n for q,n in c.items() if counts[q]>1),
            expected_acceptance=total_mass[power]/sum(candidate_counts.values()),
            expected_repeated_question_fraction=repeated_mass[power]/total_mass[power])
            for power,c in accepted.items()},
        elapsed_seconds=time.monotonic()-began,
        limitations='CPU sampling audit only; no accuracy evidence. The existing mixed sampler draws '
        'equal candidates per shard, so inverse frequency does not guarantee exactly uniform questions. '
        'All target variants remain possible. Zero means held out, never a bad program label.')
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


class FrequencyIndex:
    def __init__(self, directory, cache, *, shards=60, heldout_rows=1024):
        directory, cache = Path(directory), Path(cache)
        self.manifest = json.loads((directory/'manifest.json').read_text())
        records = self.manifest['shards']
        if self.manifest['version'] != 1 or len(records) != shards or self.manifest['heldout_rows'] != heldout_rows:
            raise ValueError('frequency index does not match training scope')
        self.arrays = []
        for shard, record in enumerate(records):
            name=f'part-{shard:05d}'
            if record['file'] != name+'.parquet' or record['index'] != name+'.npy':
                raise ValueError('unexpected index path')
            if digest(cache/record['file']) != record['sha256'] or digest(directory/record['index']) != record['index_sha256']:
                raise ValueError('frequency index or dataset checksum mismatch')
            a = np.load(directory/record['index'], mmap_mode='r', allow_pickle=False)
            if a.dtype != np.uint32 or a.shape != (record['rows'],):
                raise ValueError('invalid count array')
            self.arrays.append(a)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('Build index on CPU only')
    print(json.dumps(build(args.cache, args.output)), flush=True)
