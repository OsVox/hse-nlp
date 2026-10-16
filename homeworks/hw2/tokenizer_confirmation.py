"""Second-seed check of the cheaper code-fitted 8k tokenizer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import tokenizer_pilot
from .short_pilots import ROOT

NAME = 'tokenizer-confirm-2k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PRIOR = Path(__file__).parent/'experiments/representation-pilot-2k-v1/selection.json'
PLAN = deepcopy(tokenizer_pilot.PLAN)
PLAN.update(seed=1042, arms={'control':'original8192','code8192':'code8192'},
    prior_seed=42, confirmation_seed=1042, outer_timeout_seconds=900,
    example_presentations_per_arm=262144,
    example_budget_note='Nominal maximum; an epoch-ending partial batch can reduce this slightly and equally in both arms.',
    hypothesis='Repeat code-fitted 8k versus original 8k with a second matched initialization/data/dropout seed.',
    selection='Require strictly positive gains on both held-out subsets in both seeds, with no added total execution failures in either seed. Only then consider a bounded extension, never automatic full training.',
    limitations='Same held-out questions reused; this is a seed robustness check, not new validation or statistical significance. Early scratch learning does not establish eventual performance. Token losses across vocabularies are not comparable. Fitting sample and code weighting both differ from the original tokenizer.',
    provenance='Only original course question/code strings and randomly initialized weights. No target cleanup, extra examples, alternate targets, pretrained vocabulary or external data.')
LAUNCH = (tokenizer_pilot.LAUNCH.replace(tokenizer_pilot.NAME, NAME)
    .replace('homeworks.hw2.tokenizer_pilot','homeworks.hw2.tokenizer_confirmation')
    .replace('timeout=1600','timeout=900')
    .replace('TOKENIZER_PILOT_STARTED','TOKENIZER_CONFIRM_STARTED')
    .replace('Hard bound 1600 seconds; four scratch 2048-update arms only.',
             'Hard bound 900 seconds; two scratch 2048-update arms only.'))


def paired_selection(results, prior):
    pairs = {}
    for seed, values in ((42, prior['results']), (1042, results)):
        gains = {stage: values['code8192'][stage]['numeric_correct']-values['control'][stage]['numeric_correct']
                 for stage in ('pilot','confirmation')}
        failures = sum(values['code8192'][stage]['prediction_failures']-values['control'][stage]['prediction_failures']
                       for stage in gains)
        pairs[str(seed)] = dict(split_gains=gains,total_gain=sum(gains.values()),failure_delta=failures)
    confirmed = all(min(p['split_gains'].values())>0 and p['failure_delta']<=0 for p in pairs.values())
    return dict(results=results,paired_seeds=pairs,confirmed=confirmed,
                shortlist=['code8192'] if confirmed else [],training_updates_per_arm=2048,
                public_test_used_for_selection=False,long_run_started=False)


def select(results):
    payload=PRIOR.read_bytes()
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==ready['prior_selection_sha256']
    result=paired_selection(results,json.loads(payload))
    result.update(prior_selection_sha256=ready['prior_selection_sha256'],prior_run=tokenizer_pilot.NAME)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert hashlib.sha256(PRIOR.read_bytes()).hexdigest()==ready['prior_selection_sha256']
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
