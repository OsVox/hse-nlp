"""Bounded second-seed replication of the selected full-model Muon recipe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import optimizer_pilot, tokenizer_pilot
from .short_pilots import ROOT

NAME = 'optimizer-confirm-2k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PRIOR = Path(__file__).parent/'experiments/optimizer-scratch-2k-v1/selection.json'
PLAN = deepcopy(optimizer_pilot.PLAN)
PLAN.update(seed=1042, prior_seed=42, confirmation_seed=1042,
    arms={name:'original8192' for name in ('control','muon_low')},
    arm_options={name:deepcopy(optimizer_pilot.PLAN['arm_options'][name])
                 for name in ('control','muon_low')},
    outer_timeout_seconds=1000,
    hypothesis='Repeat selected Muon 0.001 versus AdamW 0.0003 with a different shared initialization, data and dropout seed.',
    selection='Require positive gains on both held-out subsets and no added total execution failures in BOTH seeds. Public scores never select the recipe. A pass permits a bounded learning-curve/equal-time experiment, not automatic long training.',
    limitations='Reuses validation questions; replication is not new validation or statistical significance. Equal examples do not equal GPU time. Early learning does not establish mature performance or optimal model size.',
    throughput_probe='Reuse the completed same-model native-Muon GPU probe; per-arm 480-second and whole-driver 1000-second bounds remain active.')
LAUNCH = (optimizer_pilot.LAUNCH.replace(optimizer_pilot.NAME, NAME)
    .replace('homeworks.hw2.optimizer_pilot','homeworks.hw2.optimizer_confirmation')
    .replace('timeout=1800','timeout=1000')
    .replace('OPTIMIZER_PILOT_STARTED','OPTIMIZER_CONFIRM_STARTED')
    .replace('Hard bound 1800 seconds; four fresh full-model 2048-update arms only.',
             'Hard bound 1000 seconds; two fresh full-model 2048-update arms only.'))


def paired_selection(results, prior):
    assert prior['adamw_reference']=='control'
    assert 'muon_low' in prior['shortlist']
    pairs = {}
    for seed, values in ((42,prior['results']), (1042,results)):
        gains = {stage:values['muon_low'][stage]['numeric_correct']-
                 values['control'][stage]['numeric_correct']
                 for stage in ('pilot','confirmation')}
        failures = sum(values['muon_low'][stage]['prediction_failures']-
                       values['control'][stage]['prediction_failures'] for stage in gains)
        pairs[str(seed)] = dict(split_gains=gains, failure_delta=failures)
    confirmed = all(min(v['split_gains'].values())>0 and v['failure_delta']<=0
                    for v in pairs.values())
    return dict(results=results, paired_seeds=pairs, confirmed=confirmed,
                shortlist=['muon_low'] if confirmed else [],
                training_updates_per_arm=2048, all_parameters_trained=True,
                training_from_scratch=True, public_test_used_for_selection=False,
                long_run_started=False)


def select(results):
    payload = PRIOR.read_bytes()
    ready = json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==ready['prior_selection_sha256']
    result = paired_selection(results,json.loads(payload))
    result.update(prior_selection_sha256=ready['prior_selection_sha256'],
                  prior_run=optimizer_pilot.NAME)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(PRIOR.read_bytes()).hexdigest()==ready['prior_selection_sha256']
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
