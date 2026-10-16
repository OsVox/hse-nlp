"""Second-seed confirmation of deeper decoder capacity with the Muon recipe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from . import depth_pilot, tokenizer_pilot
from .short_pilots import ROOT

NAME='depth-confirm-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PRIOR=Path(__file__).parent/'experiments/depth-muon-2k-v1/selection.json'
PLAN=deepcopy(depth_pilot.PLAN)
PLAN.update(seed=1042,prior_seed=42,confirmation_seed=1042,
    arms={n:'original8192' for n in ('control','decoder_heavy')},
    arm_options={n:deepcopy(depth_pilot.PLAN['arm_options'][n]) for n in ('control','decoder_heavy')},
    expected_parameters={n:depth_pilot.PLAN['expected_parameters'][n] for n in ('control','decoder_heavy')},
    outer_timeout_seconds=1100,
    hypothesis='Repeat the 4/6 versus 6/4 encoder/decoder comparison with seed 1042, using the same Muon recipe and nearly matched parameter counts.',
    selection='Require positive gains on both held-out subsets and no extra total execution failures in BOTH seeds. A pass permits a bounded learning-curve comparison, not an automatic long run.',
    limitations='Reuses validation questions. Two seeds do not establish statistical significance or mature-model performance. Shapes, compensating feed-forward width, and measured compute differ.',
    matched='Fresh full models; same original data, tokenizer, BF16, batch 128, 2048 updates and optimizer rates as the first depth screen. Shapes require different random tensors. G4 selected after the measured throughput comparison.')
LAUNCH=(depth_pilot.LAUNCH.replace(depth_pilot.NAME,NAME)
    .replace('homeworks.hw2.depth_pilot','homeworks.hw2.depth_confirmation')
    .replace('timeout=1400','timeout=1100')
    .replace('DEPTH_PILOT_STARTED','DEPTH_CONFIRM_STARTED')
    .replace('Hard bound 1400 seconds; three fresh depth-balance 2048-update arms only.',
             'Hard bound 1100 seconds; two fresh depth-balance 2048-update arms only.')
    .replace("    drive.mount('/content/drive', timeout_ms=180000)",
             "    assert 'RTX PRO 6000' in torch.cuda.get_device_name(), 'Requested G4 unavailable; stop before setup'\n    drive.mount('/content/drive', timeout_ms=180000)"))


def paired_selection(results,prior):
    assert 'decoder_heavy' in prior['shortlist']
    pairs={}
    for seed,values in ((42,prior['results']),(1042,results)):
        gains={s:values['decoder_heavy'][s]['numeric_correct']-values['control'][s]['numeric_correct']
               for s in ('pilot','confirmation')}
        failures=sum(values['decoder_heavy'][s]['prediction_failures']-values['control'][s]['prediction_failures'] for s in gains)
        pairs[str(seed)]=dict(split_gains=gains,failure_delta=failures)
    confirmed=all(min(v['split_gains'].values())>0 and v['failure_delta']<=0 for v in pairs.values())
    return dict(results=results,paired_seeds=pairs,confirmed=confirmed,
                shortlist=['decoder_heavy'] if confirmed else [],
                training_updates_per_arm=2048,all_parameters_trained=True,
                training_from_scratch=True,public_test_used_for_selection=False,long_run_started=False)


def select(results):
    payload=PRIOR.read_bytes();ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==ready['prior_selection_sha256']
    result=paired_selection(results,json.loads(payload))
    result.update(prior_run=depth_pilot.NAME,prior_selection_sha256=ready['prior_selection_sha256'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(PRIOR.read_bytes()).hexdigest()==ready['prior_selection_sha256']
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
