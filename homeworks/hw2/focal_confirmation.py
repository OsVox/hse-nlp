"""Confirm the focal gamma-one candidate with a second data/dropout seed."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import broad_screen, focal_screen
from .short_pilots import ROOT

NAME = 'focal-confirm-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PRIOR = Path(__file__).parent/'experiments/focal-screen-1k-v1/selection.json'
PLAN = deepcopy(focal_screen.PLAN)
PLAN.update(
    arms={'control': {'seed': 1042}, 'focal1': {'seed': 1042, 'focal_gamma': 1.0}},
    prior_seed=42, confirmation_seed=1042, outer_timeout_seconds=900,
    hypothesis='Repeat the shortlisted gamma-one loss with a second matched data/dropout seed.',
    selection='Require strictly positive gain in both held-out subsets for both seeds and no increase in total execution failures in either seed. No automatic long run.',
    limitations='Both seeds reuse the same held-out questions; this is a robustness check, not independent validation or statistical significance. Focal modulation also changes gradient scale.',
)
LAUNCH = (focal_screen.LAUNCH.replace(focal_screen.NAME, NAME)
          .replace('homeworks.hw2.focal_screen', 'homeworks.hw2.focal_confirmation')
          .replace('timeout=1200', 'timeout=900')
          .replace('FOCAL_SCREEN_STARTED', 'FOCAL_CONFIRM_STARTED')
          .replace('Hard bound 1200 seconds; three matched 1024-update arms only.',
                   'Hard bound 900 seconds; two matched 1024-update arms only.'))


def paired_selection(results, prior):
    pairs = {}
    for seed, values in ((42, prior['results']), (1042, results)):
        gains = {stage: values['focal1'][stage]['numeric_correct']-values['control'][stage]['numeric_correct']
                 for stage in ('pilot', 'confirmation')}
        failures = sum(values['focal1'][stage]['prediction_failures']-values['control'][stage]['prediction_failures']
                       for stage in gains)
        pairs[str(seed)] = dict(split_gains=gains, total_gain=sum(gains.values()), failure_delta=failures)
    confirmed = all(min(p['split_gains'].values()) > 0 and p['failure_delta'] <= 0 for p in pairs.values())
    return dict(results=results, paired_seeds=pairs, confirmed=confirmed,
                shortlist=['focal1'] if confirmed else [], training_updates_per_arm=1024,
                public_test_used_for_selection=False, long_run_started=False)


def select(results):
    payload = PRIOR.read_bytes()
    result = paired_selection(results, json.loads(payload))
    result['prior_selection_sha256'] = hashlib.sha256(payload).hexdigest()
    result['prior_run'] = focal_screen.NAME
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', required=True)
    parser.parse_args()
    broad_screen.run(prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=select)
