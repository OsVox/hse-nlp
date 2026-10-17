"""Conditional second-seed check of the final AdamW size hypothesis."""
import argparse
from copy import deepcopy
import json

from . import adamw_size_curve, learning_curve
from .checkpoint_io import sha256

NAME = 'adamw-size-confirm-v1'
PREPARED = learning_curve.ROOT/'prepared'/NAME
RUN = learning_curve.ROOT/'runs'/NAME
LAUNCH = (learning_curve.LAUNCH.replace(learning_curve.NAME, NAME)
    .replace('homeworks.hw2.learning_curve', 'homeworks.hw2.adamw_size_confirm')
    .replace('timeout=3900', 'timeout=3160')
    .replace('MATCHED_CURVE_STARTED', 'ADAMW_SIZE_CONFIRM_STARTED')
    .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
             'Hard bound3160seconds; AdamW78 then AdamW42,600training seconds each,seed1042.'))


def first_seed(prior):
    checked = adamw_size_curve.select({'adamw78': prior['results']['adamw78']},
        dict(results=prior['results']['adamw42_archived'],
             model_sha256=prior['reference_model_sha256']))
    if checked['provisional_winner'] != prior['provisional_winner']:
        raise ValueError('Prior selection disagrees with its saved measurements')
    return checked


def build_plan(prior):
    if not first_seed(prior)['second_seed_required']:
        raise ValueError('AdamW78 did not pass; do not allocate a confirmation run')
    plan = deepcopy(learning_curve.PLAN)
    plan['options']['seed'] = 1042
    plan['arms'] = {'adamw78': deepcopy(adamw_size_curve.PLAN['arms']['adamw78']),
                    'adamw42': deepcopy(learning_curve.CONFIGS['adamw42'])}
    plan.update(outer_timeout_seconds=3160, prior_seed=42, confirmation_seed=1042,
        hypothesis='Confirm the AdamW78 early learning advantage in a fresh paired run against AdamW42.',
        selection='Require strict held-out gains in both subsets with no added total execution failures in both seeds. Otherwise retain AdamW42. Close the size screen after this test.',
        reference='Fresh seed1042 models, larger arm first, same GPU/runtime, original tokenizer, complete data stream, 600 measured seconds each, and greedy evaluator.',
        limitations='The first-seed baseline was archived. This paired second seed reduces host-load confounding but short-run rankings do not prove long-run optimality.',
        second_seed_required_for_promotion=True, additional_sweep_authorized=False)
    return plan


def paired_selection(results, prior):
    build_plan(prior)
    if set(results) != {'adamw42', 'adamw78'}:
        raise ValueError('Confirmation requires exactly the two AdamW sizes')
    checked = adamw_size_curve.select({'adamw78': results['adamw78']},
        dict(results=results['adamw42'],model_sha256='fresh-second-seed-control'))
    confirmed = checked['second_seed_required']
    return dict(results=results, candidate='adamw78', confirmed=confirmed,
        paired_seeds={'42': first_seed(prior)['comparison'], '1042': checked['comparison']},
        full_pass_recipe='adamw78' if confirmed else 'adamw42',
        public_test_used_for_selection=False, long_run_started=False,
        screen_closed=True, additional_sweep_authorized=False)


def run():
    ready = json.loads((PREPARED/'ready.json').read_text())
    prior_path = PREPARED/'prior-selection.json'
    assert sha256(prior_path) == ready['prior_selection_sha256']
    prior = json.loads(prior_path.read_text())
    plan = build_plan(prior)
    assert plan == ready['plan']
    learning_curve.run(prepared=PREPARED, run_dir=RUN, plan=plan,
        selection_fn=lambda results: paired_selection(results, prior))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', required=True)
    parser.parse_args()
    run()
