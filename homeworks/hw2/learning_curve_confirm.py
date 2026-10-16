"""Second-seed confirmation of the predeclared matched-time winner only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from . import learning_curve
from .checkpoint_io import sha256
from .short_pilots import ROOT

NAME='matched-curve-confirm-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PRIOR=Path(__file__).parent/'experiments/matched-curve-v1/selection.json'
LAUNCH=(learning_curve.LAUNCH.replace(learning_curve.NAME,NAME)
    .replace('homeworks.hw2.learning_curve','homeworks.hw2.learning_curve_confirm')
    .replace('timeout=3900','timeout=2800')
    .replace('MATCHED_CURVE_STARTED','MATCHED_CONFIRM_STARTED')
    .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
             'Hard bound2800seconds; two arms,600training seconds each, two checkpoints per arm.'))


def build_plan(prior):
    computed=learning_curve.select(prior['results'])
    if computed['provisional_winner']!=prior['provisional_winner']:
        raise ValueError('Prior selection disagrees with its saved results')
    candidate=computed['provisional_winner']
    if candidate=='adamw42':
        raise ValueError('No alternative passed; no confirmation GPU run is needed')
    plan=deepcopy(learning_curve.PLAN)
    plan['options']['seed']=1042
    plan['arms']={k:deepcopy(learning_curve.CONFIGS[k]) for k in (candidate,'adamw42')}
    plan.update(outer_timeout_seconds=2800,prior_seed=42,confirmation_seed=1042,candidate=candidate,
        hypothesis='Confirm only the held-out selected alternative against AdamW42 with seed1042 and reversed arm order, at the same600-second training budget.',
        selection='The alternative must improve both fresh held-out subsets without extra execution failures in both seeds. If it fails, retain AdamW42 as the reference for the authorized full pass. No public score selects the recipe.',
        reference='Same options, original examples and complete-sequence pipeline as the first comparison. Reverse arm order and use seed1042. Preserve the full-pass schedule. No new sweep or size-specific tuning.')
    return plan


def paired_selection(results, prior):
    candidate=build_plan(prior)['candidate']
    if set(results)!={candidate,'adamw42'}:
        raise ValueError('Confirmation must contain exactly the selected candidate and control')
    comparisons={str(seed):learning_curve.comparison(values[candidate],values['adamw42'])
        for seed,values in ((42,prior['results']),(1042,results))}
    confirmed=all(r['passes'] for r in comparisons.values())
    return dict(results=results,paired_seeds=comparisons,candidate=candidate,
        confirmed=confirmed,full_pass_recipe=candidate if confirmed else 'adamw42',
        public_test_used_for_selection=False,long_run_started=False,
        full_pass_completed=False,additional_sweep_authorized=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert sha256(PRIOR)==ready['prior_selection_sha256']
    prior=json.loads(PRIOR.read_text());plan=build_plan(prior)
    assert plan==ready['plan']
    learning_curve.run(prepared=PREPARED,run_dir=RUN,plan=plan,
        selection_fn=lambda results:paired_selection(results,prior))
