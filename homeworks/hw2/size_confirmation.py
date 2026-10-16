"""Second-seed replication of the shortlisted small scratch Transformer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import size_pilot, tokenizer_pilot
from .short_pilots import ROOT

NAME = 'size-confirm-2k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PRIOR = Path(__file__).parent/'experiments/size-muon-2k-v1/selection.json'
CONFIGS = {n:deepcopy(size_pilot.CONFIGS[n]) for n in ('control','small')}
PLAN = deepcopy(size_pilot.PLAN)
PLAN.update(seed=1042,prior_seed=42,confirmation_seed=1042,
    arms={n:'original8192' for n in CONFIGS},
    arm_options={n:deepcopy(size_pilot.PLAN['arm_options'][n]) for n in CONFIGS},
    expected_parameters={n:size_pilot.PLAN['expected_parameters'][n] for n in CONFIGS},
    outer_timeout_seconds=1100,
    hypothesis='Replicate the 17.58M versus 42.48M early-learning comparison with a second seed before choosing longer-run candidates.',
    matched='Same original tokenizer, Muon rates, capped source/target lengths, mixed sampler, batch128 and 2048 updates as the first size screen; seed1042. Reverse arm order. This isolates replication rather than combining the later data-pipeline changes.',
    selection='The smaller model must improve both held-out subsets with no extra total execution failures in both seeds. Public scores do not select the recipe. A pass advances it to matched-compute learning-curve confirmation, not directly to full training.',
    limitations='Reused held-out questions and short capped runs are exploratory. This replication does not establish complete-example exposure or an optimal model size. All longer recipe comparisons must use the verified full-coverage complete-sequence pipeline.',
    next_stages=['Complete bounded matched-compute learning curves on newly reserved questions with full original sequences.',
                 'Then execute the authorized longer run using the best confirmed recipe and exact full-pass accounting.'])
LAUNCH = (size_pilot.LAUNCH.replace(size_pilot.NAME,NAME)
    .replace('homeworks.hw2.size_pilot','homeworks.hw2.size_confirmation')
    .replace('timeout=1400','timeout=1100')
    .replace('SIZE_PILOT_STARTED','SIZE_CONFIRM_STARTED')
    .replace('Hard bound 1400 seconds; three fresh width/capacity 2048-update arms only.',
             'Hard bound 1100 seconds; two fresh size-confirmation 2048-update arms only.'))


def paired_selection(results, prior):
    assert 'small' in prior['shortlist']
    pairs = {}
    for seed, values in ((42,prior['results']),(1042,results)):
        gains={s:values['small'][s]['numeric_correct']-values['control'][s]['numeric_correct']
               for s in ('pilot','confirmation')}
        failures=sum(values['small'][s]['prediction_failures']-values['control'][s]['prediction_failures'] for s in gains)
        pairs[str(seed)] = dict(split_gains=gains,failure_delta=failures)
    confirmed=all(min(v['split_gains'].values())>0 and v['failure_delta']<=0 for v in pairs.values())
    return dict(results=results,paired_seeds=pairs,confirmed=confirmed,
                shortlist=['small'] if confirmed else [],training_updates_per_arm=2048,
                all_parameters_trained=True,training_from_scratch=True,
                public_test_used_for_selection=False,long_run_started=False,
                equal_gpu_time=False,optimal_model_size_established=False)


def select(results):
    ready=json.loads((PREPARED/'ready.json').read_text())
    payload=PRIOR.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['prior_selection_sha256']
    return {**paired_selection(results,json.loads(payload)),
            'prior_run':size_pilot.NAME,'prior_selection_sha256':ready['prior_selection_sha256']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(PRIOR.read_bytes()).hexdigest()==ready['prior_selection_sha256']
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
