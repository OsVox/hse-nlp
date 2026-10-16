"""CPU-only bounds on training coverage; never equate presentations with unique rows.

Run from the repository root with python -m homeworks.hw2.coverage_readiness_audit.
This reads archived evidence, not model weights, and does not allocate a runtime.
"""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'homeworks/hw2/experiments'


def lineage_bound(config, end_step):
    """Use actual child start steps, not a parent's possibly unfinished budget."""
    phases = []
    while config:
        start = config.get('schedule_start') or 0
        if not 0 <= start < end_step:
            raise ValueError('Nonmonotone checkpoint lineage')
        batch = config['batch_size']
        accumulation = config['accumulation']
        phases.append(dict(start_step=start, end_step=end_step,
            batch_size=batch, accumulation=accumulation,
            presentation_upper_bound=(end_step-start)*batch*accumulation,
            max_shards=config['max_shards'],
            mixed_rows_per_shard=config.get('mixed_rows_per_shard')))
        parent = config.get('parent_checkpoint')
        if start and not parent:
            raise ValueError('Missing parent; cannot establish whole-lineage bound')
        config = parent['run_config'] if parent else None
        end_step = start
    return list(reversed(phases))


def audit():
    inputs = {
        'corpus': EXPERIMENTS/'train-diversity-audit-v1/results.json',
        'large_budget': EXPERIMENTS/'size-muon-2k-v1/large/compute-budget.json',
        'large_config': EXPERIMENTS/'size-muon-2k-v1/large/step-002048/run_config.json',
        'best_config': EXPERIMENTS/'frequency-screen-1k-v1/control/step-073024/run_config.json',
        'best_step': EXPERIMENTS/'frequency-screen-1k-v1/control/step-073024/step.json',
        'loader': ROOT/'homeworks/hw2/data.py',
        'trainer': ROOT/'homeworks/hw2/train.py',
    }
    read = lambda key: json.loads(inputs[key].read_text())
    corpus, budget = read('corpus'), read('large_budget')
    eligible = sum(s['retained'] for s in corpus['shards'])
    assert len(corpus['shards']) == 60
    assert eligible == corpus['total_retained_rows']
    large = read('large_config')
    assert large['max_shards'] == 60 and large['mixed_rows_per_shard'] == 4096
    phases = lineage_bound(read('best_config'), read('best_step')['step'])
    best_bound = sum(p['presentation_upper_bound'] for p in phases)
    # These are upper bounds even if every presentation had a distinct row ID.
    assert budget['examples_processed'] < eligible and best_bound < eligible
    return {
        'date': '2026-10-16',
        'gpu_allocated_by_audit': False,
        'scaling_ready': False,
        'source_files': {k: {'path': str(p.relative_to(ROOT)),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in inputs.items()},
        'eligible_rows': eligible,
        'raw_rows': sum(s['rows'] for s in corpus['shards']),
        'excluded_rows': sum(s['excluded'] for s in corpus['shards']),
        'distinct_exact_question_hashes': corpus['unique_question_hashes'],
        'all_shards_available_to_sampler': True,
        'all_eligible_rows_seen_by_large_model': False,
        'all_eligible_rows_seen_by_best_model': False,
        'large_78m': {
            'updates': budget['updates'],
            'actual_presentations': budget['examples_processed'],
            'distinct_row_coverage_upper_bound_percent': 100*budget['examples_processed']/eligible,
            'exact_distinct_rows_consumed': None,
        },
        'best_42m': {
            'phases': phases,
            'presentation_upper_bound': best_bound,
            'distinct_row_coverage_upper_bound_percent': 100*best_bound/eligible,
            'exact_distinct_rows_consumed': None,
        },
        'minimum_updates_for_one_pass_at_batch128': math.ceil(eligible/128),
        'limitations': [
            'Corpus counts reuse the checksum-recorded full CPU scan; this audit does not rescan all Parquet bytes.',
            'Presentation bounds are not measured unique-row coverage. Resampling and partial batches lower coverage.',
            'The old loader samples without replacement within each shard-round but allows repeats across rounds.',
            'Resuming changes the sampling seed; there is no saved consumed-row cursor.',
            'Other independent experiments do not contribute experience to either checkpoint.',
            'A full pass means all eligible original rows; held-out questions must remain excluded.',
            'Some visited rows have truncated inputs or targets; row coverage does not prove full-token exposure.',
        ],
        'required_before_scaling': [
            'Implement a globally mixed traversal without repeating row IDs until all eligible rows have been visited.',
            'Carry original shard and row IDs through grouping, and count only batches used by optimizer updates.',
            'Persist consumed-stream state with checkpoints; do not count prefetched batches as trained.',
            'CPU tests must cover uneven shards, partial batches, held-out exclusion, end-of-epoch and mid-window resume.',
            'Benchmark the corrected loader against current throughput before allocating a long GPU run.',
            'Report per-shard unique rows, repeats, exclusions and completed full passes at every checkpoint.',
        ],
        'optimization_claim': 'Not exhausted. Completed short screens only apply to their tested settings and budgets.',
    }


if __name__ == '__main__':
    destination = EXPERIMENTS/'coverage-readiness-cpu-v1'
    destination.mkdir(exist_ok=True)
    result = audit()
    (destination/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['eligible_rows', 'large_78m',
        'minimum_updates_for_one_pass_at_batch128', 'scaling_ready']}, indent=2))
