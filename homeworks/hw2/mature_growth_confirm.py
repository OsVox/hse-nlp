"""Confirm mature FFN growth with new added weights and dropout randomness.

Both arms retain the same learned parent, optimizer history, schedule and row
cursor. This is a continuation replicate, not an independent scratch model.
"""
import argparse
import json
from pathlib import Path

from . import mature_growth_pilot as pilot
from .checkpoint_io import sha256
from .prepare_mature_growth import plan_from_parent
from .short_pilots import ROOT

NAME = 'mature-growth-confirm-300s-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PRIOR = Path(__file__).parent/'experiments/mature-growth-300s-v1/selection.json'
LAUNCH = (pilot.LAUNCH.replace(pilot.NAME, NAME)
    .replace('homeworks.hw2.mature_growth_pilot', 'homeworks.hw2.mature_growth_confirm')
    .replace('MATURE_GROWTH_STARTED', 'MATURE_GROWTH_CONFIRM_STARTED'))


def build_plan(config, step, model_hash):
    plan = plan_from_parent(config, step, model_hash)
    plan.update(name=NAME, arms=['grown66', 'control42'], growth_seed=1042,
        continuation_seed=1042,
        hypothesis='Confirm the first paired width gain with fresh added-weight and dropout seeds; reverse arm order. The learned parent and original data cursor remain shared.',
        selection='Require the grown model to beat its same-replicate control in both held-out subsets, with no added total prediction failures, in both repetitions. Report changes versus the mature parent separately.',
        long_training_requires='Both paired repetitions must qualify. Then bounded continuation with validation stopping is authorized by the user; no new architecture sweep.',
        independence_limit='New growth and dropout randomness, not independently trained parents or a new held-out set.')
    return plan


def select(results, parent, prior):
    first = pilot.select(prior['results'], parent)
    if not first['paired_comparison']['passes']:
        raise ValueError('No qualifying first repetition')
    second = pilot.select(results, parent)
    confirmed = second['paired_comparison']['passes']
    return dict(second, confirmed=confirmed,
        paired_repetitions={'42': first['paired_comparison'], '1042': second['paired_comparison']},
        scale_recipe='grown66' if confirmed else None,
        confirmation_required=False,
        next_action='Bounded grown66 continuation with validation stopping' if confirmed else 'Retain scored best; no long width run justified',
        independence_limit='Shared learned parent and held-out questions; only added-weight/dropout seeds and arm order differ.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true', required=True)
    p.parse_args()
    ready = json.loads((PREPARED/'ready.json').read_text())
    assert sha256(PRIOR) == ready['prior_selection_sha256']
    prior = json.loads(PRIOR.read_text())
    pilot.run(prepared=PREPARED, run_dir=RUN, plan_fn=build_plan,
        selection_fn=lambda values, parent: select(values, parent, prior))
