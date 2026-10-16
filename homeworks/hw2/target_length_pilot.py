"""Isolate original-target retention with identical consumed rows and decoding."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tarfile

from . import broad_screen, coverage_pilot
from .checkpoint_io import sha256
from .short_pilots import ROOT, copy_verified

NAME='target-length-1k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
INDEX=Path('/content/target-length-inputs/coverage-index')
PLAN=deepcopy(coverage_pilot.PLAN)
PLAN.update(
    arms={name:dict(numeric_token_weight=2.,mixed_rows_per_shard=None,
                   coverage_index=str(INDEX),training_target_length=cap)
          for name,cap in [('control',192),('target384',384)]},
    hypothesis='Retaining more of each original program may improve learning compared with training on truncated programs.',
    matched='Same own 42M parent weights and AdamW moments, 1024 updates, batch128, BF16, seed42, numeric weight2, warmup50 and rate3e-5. Source cap128; grouping target cap192 in both arms preserves exact rows/order/batches. Only training target cap changes 192 to384. Validation and greedy generation cap192 remain fixed.',
    selection='Require positive executed-answer gain on both held-out subsets, no extra total execution failures, and no score loss against parent on either subset. Repeat any survivor with another seed before combining or extending. Verify identical row-order hashes and 131072 unique consumed rows in each arm. Public score does not select the recipe.',
    limitations='Short mature-model continuation and reused held-out sets. Equal updates differ in compute and retained tokens. Inference cap192 can limit long-program accuracy; test that separately. 384 still truncates 1184 eligible programs. No complete training pass or automatic long run.',
    next_stages=['Verify CPU correctness and fixed row order before GPU.','Archive and bot-score every retained checkpoint.','Only repeat a held-out survivor; otherwise record the negative result without ruling out larger-budget target retention.'])
LAUNCH=(coverage_pilot.LAUNCH.replace(coverage_pilot.NAME,NAME)
    .replace('homeworks.hw2.coverage_pilot','homeworks.hw2.target_length_pilot')
    .replace('COVERAGE_PIPELINE_STARTED','TARGET_LENGTH_STARTED'))


def restore_index(run_dir):
    ready=json.loads((PREPARED/'ready.json').read_text())
    archive=INDEX.parent/'index.tar.gz'
    archive.parent.mkdir(exist_ok=False)
    copy_verified(Path(ready['coverage_index_archive_path']),archive)
    assert sha256(archive)==ready['coverage_index_sha256']
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent,filter='data')
    assert sha256(INDEX/'manifest.json')==ready['coverage_manifest_sha256']
    (run_dir/'index-verification.json').write_text(json.dumps(dict(
        archive_sha256=ready['coverage_index_sha256'],
        manifest_sha256=ready['coverage_manifest_sha256']),indent=2))


def select(results):
    receipts={}
    for arm,cap in [('control',192),('target384',384)]:
        record=json.loads((RUN/arm/f"step-{PLAN['final_step']:06d}"/'coverage.json').read_text())
        r=record['report']
        assert record['optimizer_step']==PLAN['final_step']
        assert r['presentations']==r['unique_rows_since_tracking_started']==131072
        assert r['repeated_presentations_since_tracking_started']==r['completed_full_passes']==0
        assert r['start_step']==PLAN['parent_step'] and r['target_length']==cap
        assert r['source_length']==128 and r['grouping_target_length']==192
        receipts[arm]=record
    assert receipts['control']['state']['row_order_sha256']==receipts['target384']['state']['row_order_sha256']
    gains={s:results['target384'][s]['numeric_correct']-results['control'][s]['numeric_correct']
           for s in ('pilot','confirmation')}
    parent=json.loads((PREPARED/'ready.json').read_text())['parent_heldout']
    parent_gains={s:results['target384'][s]['numeric_correct']-parent[s]['numeric_correct'] for s in gains}
    failures=sum(results['target384'][s]['prediction_failures']-results['control'][s]['prediction_failures'] for s in gains)
    eligible=min(gains.values())>0 and min(parent_gains.values())>=0 and failures<=0
    return dict(results=results,split_gains=gains,parent_gains=parent_gains,failure_delta=failures,
        shortlist=['target384'] if eligible else [],coverage_receipts=receipts,
        exact_row_order_matched=True,training_updates_per_arm=1024,all_parameters_trained=True,
        public_test_used_for_selection=False,long_run_started=False,second_seed_required=True,
        source_cap=128,validation_target_cap=192,generation_cap=192,grouping_target_cap=192)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select,
                     runtime_preflight_fn=restore_index)
