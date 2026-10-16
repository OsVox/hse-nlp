"""Matched scratch SwiGLU/GEGLU screen with exact consumed-row accounting."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tarfile

from . import size_pilot, tokenizer_pilot
from .checkpoint_io import sha256
from .short_pilots import ROOT, copy_verified

NAME='activation-muon-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
INDEX=Path('/content/activation-inputs/coverage-index')
ACTIVATIONS={'control':'gated-gelu','swiglu':'gated-silu'}
PLAN=deepcopy(size_pilot.PLAN)
PLAN.update(arms={name:'original8192' for name in ACTIVATIONS},
    mixed_rows_per_shard=None,require_identical_initialization=True,
    arm_options={name:dict(optimizer='muon',learning_rate=.0003,muon_learning_rate=.001,
        feed_forward_proj=act,coverage_index=str(INDEX)) for name,act in ACTIVATIONS.items()},
    expected_parameters={name:42481152 for name in ACTIVATIONS},
    outer_timeout_seconds=1000,training_timeout_seconds=480,
    hypothesis='Does replacing gated-GELU with SwiGLU improve early executed-answer learning at identical parameter count?',
    matched='Identical full random weights, seed42, original tokenizer, full-coverage row order, batch128, 2048 updates, BF16, 200-step warmup and cosine schedule. Native Muon0.001 on hidden matrices and AdamW0.0003 on embeddings/vectors. Source/target caps128/192 and greedy decoding stay fixed. Only the gated feed-forward activation changes.',
    selection='Require positive executed-answer gains on BOTH held-out subsets and no extra total execution failures. Verify identical full initialization and consumed-row hashes, zero repeats and 262144 consumed rows in each arm. Repeat any survivor with another seed before combining or extending. Public scores are recorded separately.',
    limitations='Short scratch pilot and reused held-out sets; equal updates do not ensure equal GPU time. Rates are not separately tuned. The existing caps still truncate some original text; this isolated activation test does not resolve that data issue or complete a training pass. No automatic long run.',
    reference='https://arxiv.org/abs/2002.05202',
    next_stages=['Repeat a surviving activation with a second seed.','Resolve long original examples without silent truncation.','Require a verified full-pass training receipt before claiming all eligible examples were trained.'])
LAUNCH=(size_pilot.LAUNCH.replace(size_pilot.NAME,NAME)
    .replace('homeworks.hw2.size_pilot','homeworks.hw2.activation_pilot')
    .replace('timeout=1400','timeout=1000')
    .replace('SIZE_PILOT_STARTED','ACTIVATION_PILOT_STARTED')
    .replace('Hard bound 1400 seconds; three fresh width/capacity 2048-update arms only.',
             'Hard bound 1000 seconds; two fresh activation 2048-update arms only.'))


def restore_index(run_dir):
    ready=json.loads((PREPARED/'ready.json').read_text())
    archive=INDEX.parent/'index.tar.gz';archive.parent.mkdir(exist_ok=False)
    copy_verified(Path(ready['coverage_index_archive_path']),archive)
    assert sha256(archive)==ready['coverage_index_sha256']
    with tarfile.open(archive) as tar:tar.extractall(archive.parent,filter='data')
    assert sha256(INDEX/'manifest.json')==ready['coverage_manifest_sha256']
    (run_dir/'index-verification.json').write_text(json.dumps(dict(
        archive_sha256=ready['coverage_index_sha256'],manifest_sha256=ready['coverage_manifest_sha256']),indent=2))


def quality_selection(results):
    gains={s:results['swiglu'][s]['numeric_correct']-results['control'][s]['numeric_correct']
           for s in ('pilot','confirmation')}
    failures=sum(results['swiglu'][s]['prediction_failures']-results['control'][s]['prediction_failures'] for s in gains)
    return dict(results=results,split_gains=gains,failure_delta=failures,
        shortlist=['swiglu'] if min(gains.values())>0 and failures<=0 else [],
        training_from_scratch=True,all_parameters_trained=True,training_updates_per_arm=2048,
        public_test_used_for_selection=False,long_run_started=False,second_seed_required=True,
        equal_gpu_time=False)


def verify_records(records):
    for arm,activation in ACTIVATIONS.items():
        coverage,config=records[arm]['coverage'],records[arm]['config']
        report=coverage['report']
        assert coverage['optimizer_step']==2048
        assert report['presentations']==report['unique_rows_since_tracking_started']==262144
        assert report['repeated_presentations_since_tracking_started']==report['completed_full_passes']==0
        assert report['start_step']==0 and report['source_length']==128 and report['target_length']==192
        assert config['model_config']['feed_forward_proj']==activation
        assert config['initial_model_sha256'] and len(config['initial_model_sha256'])==64
    assert records['control']['config']['initial_model_sha256']==records['swiglu']['config']['initial_model_sha256']
    assert records['control']['coverage']['state']['row_order_sha256']==records['swiglu']['coverage']['state']['row_order_sha256']
    return dict(exact_initial_weights_matched=True,exact_row_order_matched=True,
                coverage_receipts={n:v['coverage'] for n,v in records.items()},
                initial_model_sha256=records['control']['config']['initial_model_sha256'])


def select(results):
    records={name:dict(coverage=json.loads((RUN/name/'step-002048/coverage.json').read_text()),
                      config=json.loads((RUN/name/'step-002048/run_config.json').read_text()))
             for name in ACTIVATIONS}
    return {**quality_selection(results),**verify_records(records)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select,
                        runtime_preflight_fn=restore_index)
