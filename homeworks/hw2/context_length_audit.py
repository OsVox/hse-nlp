"""Measure original question lengths on CPU without changing model inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length_summary(lengths):
    a = np.asarray(lengths, dtype=np.int64)
    assert len(a) and (a > 0).all()
    return {
        'rows': len(a), 'mean_tokens': float(a.mean()),
        'quantiles': {str(q): int(np.quantile(a, q, method='higher'))
                      for q in [.5, .9, .95, .99, 1.]},
        'caps': {str(cap): {
            'questions_cut': int((a > cap).sum()),
            'questions_cut_percent': float(100 * (a > cap).mean()),
            'removed_tokens': int(np.maximum(a - cap, 0).sum()),
        } for cap in [64, 80, 96, 112, 128, 160, 192]},
        'histogram': {str(n): int(c) for n, c in zip(*np.unique(a, return_counts=True))},
    }


def lengths(tokenizer, texts):
    result = []
    for start in range(0, len(texts), 1024):
        result.extend(len(e.ids) for e in tokenizer.encode_batch(texts[start:start + 1024]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('homeworks/hw2/cache'))
    p.add_argument('--tokenizer', type=Path, default=Path('homeworks/hw2/tokenizer/tokenizer.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    began = time.monotonic()
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_padding()
    tok.no_truncation()
    shard = args.cache / 'part-00000.parquet'
    test = args.cache / 'test.parquet'
    all_train = pq.read_table(shard, columns=['question'])['question'].to_pylist()
    heldout = set(all_train[:1024])
    train = [q for q in all_train[1024:] if q not in heldout]
    test_questions = pq.read_table(test, columns=['question'])['question'].to_pylist()
    assert len(test_questions) == 1919
    result = dict(
        tokenizer_sha256=sha256(args.tokenizer),
        input_sha256={'train_shard_0': sha256(shard), 'test': sha256(test)},
        token_count_includes_eos=True, padding=False, truncation=False,
        train_scope='All eligible rows of shard 0; exact held-out question texts excluded. Not the full 60-shard training corpus.',
        test_scope='All official test questions, preserving original order. Questions only; no answers used.',
        source_cap_unchanged=128, target_cap_unchanged=192,
        training_changed=False, inference_changed=False, gpu_used=False,
        results={'train_shard_0': length_summary(lengths(tok, train)),
                 'official_test': length_summary(lengths(tok, test_questions))},
    )
    prior = Path('homeworks/hw2/experiments/cpu-data-tokenizer-audit-v1/tokenizers/results.json')
    if prior.exists():
        previous = json.loads(prior.read_text())['results']['original8192']
        assert previous['tokenizer_sha256'] == result['tokenizer_sha256']
        result['prior_60_shard_sample'] = {'source': str(prior), 'sha256': sha256(prior),
                                          'question': previous['question'],
                                          'scope': '15,304 distinct questions sampled across 60 shards in earlier CPU audit.'}
    result['elapsed_seconds'] = time.monotonic() - began
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'lengths.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {n: v for n, v in s.items() if n != 'histogram'}
                      for k, s in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
