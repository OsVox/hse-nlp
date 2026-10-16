"""CPU-only original-training-data audit; does not produce training targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def question_key(question):
    return hashlib.blake2b(question.encode(), digest_size=16).digest()


def summary(values):
    values = sorted(values)
    if not values:
        return {}
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p90=values[int(.9 * (len(values) - 1))],
                p99=values[int(.99 * (len(values) - 1))], max=max(values))


def audit(cache, tokenizer_path, output, *, max_shards=60, sample_per_shard=2048):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heldout = set(pq.read_table(cache/'part-00000.parquet', columns=['question'])
                  .slice(0, 1024)['question'].to_pylist())
    counts = Counter()
    samples = []
    shards = []
    for shard in range(max_shards):
        path = cache/f'part-{shard:05d}.parquet'
        table = pq.read_table(path, columns=['question', 'code'])
        questions = table['question'].to_pylist()
        allowed = [i for i, question in enumerate(questions) if question not in heldout]
        counts.update(question_key(questions[i]) for i in allowed)
        rng = random.Random(20261016 + shard)
        indices = rng.sample(allowed, min(sample_per_shard, len(allowed)))
        samples.extend(table.take(indices).to_pylist())
        shards.append(dict(shard=shard, rows=len(table), retained=len(allowed),
                           sampled=len(indices), excluded=len(table)-len(allowed),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (output/'progress.json').write_text(json.dumps(dict(shards_completed=shard+1,
            distinct_question_hashes=len(counts), elapsed_seconds=time.monotonic()-started)))
        print('shard', shard, 'unique_questions', len(counts), flush=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    sources, targets, comments = [], [], []
    for start in range(0, len(samples), 1024):
        chunk = samples[start:start+1024]
        sources.extend(len(x.ids) for x in tokenizer.encode_batch([r['question'] for r in chunk]))
        targets.extend(len(x.ids) for x in tokenizer.encode_batch([r['code'] for r in chunk]))
        # A text statistic only; '#' inside a string is not a Python comment.
        comments.extend(sum(len(line.partition('#')[2]) for line in r['code'].splitlines()) /
                        max(1, len(r['code'])) for r in chunk)
    histogram = Counter(counts.values())
    total = sum(counts.values())
    unique = len(counts)
    sample_repetitions = [counts[question_key(r['question'])] for r in samples]
    result = dict(
        purpose='Audit original training distribution without creating or changing targets',
        shards=shards, total_retained_rows=total, unique_question_hashes=unique,
        question_key='BLAKE2b-128 of exact UTF-8 question; negligible hash-collision risk',
        excess_duplicate_row_fraction=(total-unique)/total,
        rows_from_repeated_questions_fraction=sum(n*c for n,c in histogram.items() if n>1)/total,
        question_frequency_histogram=dict(sorted(histogram.items())),
        question_frequency_summary=summary(list(counts.values())),
        sample_repetition_summary=summary(sample_repetitions),
        sample_rows=len(samples), source_tokens=summary(sources), target_tokens=summary(targets),
        source_over_128=sum(x>128 for x in sources), target_over_192=sum(x>192 for x in targets),
        text_after_hash_character_fraction=summary(comments),
        token_length_bins={f'{lo}:{hi}':sum(lo<=n<hi for n in targets)
                           for lo,hi in [(0,64),(64,96),(96,128),(128,192),(192,384),(384,100000)]},
        elapsed_seconds=time.monotonic()-started,
        exclusions='All copies of the 1024 held-out question texts, across all scanned shards',
        limitations='Exact string groups only. Length and comment statistics use a fixed random sample. '
                    'Does not measure semantic correctness or inspect test questions.',
    )
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['shards','question_frequency_histogram']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-shards', type=int, default=60)
    parser.add_argument('--sample-per-shard', type=int, default=2048)
    args = parser.parse_args()
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('This audit must run without a GPU allocation')
    audit(args.cache, args.tokenizer, args.output, max_shards=args.max_shards,
          sample_per_shard=args.sample_per_shard)
