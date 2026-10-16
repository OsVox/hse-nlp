"""Bounded GPU/quality regression check of the CPU-verified full-pass pipeline."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tarfile

from . import broad_screen, muon_transfer
from .checkpoint_io import sha256
from .short_pilots import ROOT, CACHE, copy_verified

NAME='coverage-pipeline-1k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
INDEX=Path('/content/coverage-screen-inputs/coverage-index')
PLAN=deepcopy(muon_transfer.PLAN)
PLAN.update(outer_timeout_seconds=900,
    arms={'control':{'numeric_token_weight':2.},
          'coverage':{'numeric_token_weight':2.,'mixed_rows_per_shard':None,
                      'coverage_index':str(INDEX)}},
    hypothesis='Verify the full-pass pipeline on the actual GPU without an obvious quality or throughput regression.',
    matched='Same best own 42M weights and AdamW moments, 1024 updates, batch128, BF16, seed42, numeric-token weight2, warmup50 and peak rate3e-5. Only row traversal changes; sampled examples and resulting padding can differ.',
    selection='Correctness requires 131072 tracked presentations, zero repeated row IDs within this new traversal, no completed pass claimed, and correct optimizer-step binding. Investigate held-out losses larger than 5/245 or 10/493, or more than five added execution failures. This is an engineering regression screen, not a significance test or evidence for an accuracy gain.',
    limitations='Both 1024-update arms are shorter than one sampling round. They cannot measure the long-term benefit of removing cross-round repeats. Old unique coverage remains unknown. Validation is reused. No automatic long run.',
    next_stages=['Inspect measured GPU cost and exact coverage receipt.','Archive and bot-score both checkpoints.','Keep longer training on hold while checking remaining prioritized optimizations.'])
LAUNCH=(muon_transfer.LAUNCH.replace(muon_transfer.NAME,NAME)
    .replace("    assert hasattr(torch.optim, 'Muon')\n",'')
    .replace('homeworks.hw2.muon_transfer','homeworks.hw2.coverage_pilot')
    .replace('timeout=1200','timeout=900')
    .replace('MUON_TRANSFER_STARTED','COVERAGE_PIPELINE_STARTED')
    .replace('Hard bound 1200 seconds; four matched 1024-update arms only.',
             'Hard bound 900 seconds; two matched 1024-update arms only.'))


def restore_index(run_dir):
    ready=json.loads((PREPARED/'ready.json').read_text())
    archive=Path('/content/coverage-screen-inputs/index.tar.gz')
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
    gains={s:results['coverage'][s]['numeric_correct']-results['control'][s]['numeric_correct']
           for s in ['pilot','confirmation']}
    failures=sum(results['coverage'][s]['prediction_failures']-
                 results['control'][s]['prediction_failures'] for s in gains)
    checkpoint=RUN/'coverage'/f"step-{PLAN['final_step']:06d}"
    coverage=json.loads((checkpoint/'coverage.json').read_text())
    report=coverage['report']
    assert coverage['optimizer_step']==PLAN['final_step']
    assert report['presentations']==131072
    assert report['unique_rows_since_tracking_started']==131072
    assert report['repeated_presentations_since_tracking_started']==0
    assert report['completed_full_passes']==0 and report['start_step']==PLAN['parent_step']
    return dict(results=results,split_gains=gains,failure_delta=failures,
        coverage_receipt=coverage,coverage_correctness_passed=True,
        material_quality_regression=gains['pilot'] < -5 or gains['confirmation'] < -10 or failures > 5,
        training_updates_per_arm=1024,all_parameters_trained=True,
        public_test_used_for_selection=False,long_run_started=False,
        long_term_coverage_benefit_measured=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select,
                     runtime_preflight_fn=restore_index)
