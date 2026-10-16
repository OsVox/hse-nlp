"""Two-seed, example-matched test of within-window grouped batch order."""
import argparse
from copy import deepcopy

from . import broad_screen
from .short_pilots import ROOT

NAME = 'batch-order-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PLAN = deepcopy(broad_screen.PLAN)
PLAN.update(
    final_step=71024, save_every=71024, eval_every=71024,
    added_example_presentations_per_arm=131072,
    training_updates_per_arm=1024, outer_timeout_seconds=1200,
    arms={f'{order}_s{seed}': dict(seed=seed, **({'shuffle_batch_order': True} if order=='shuffled' else {}))
          for seed in (42,1042) for order in ('control','shuffled')},
    hypothesis='Preserve exact examples, batches and padding cost; shuffle only the ascending-length batch sequence.',
    selection='Require positive total gain in each seed, nonnegative gain in each subset in each seed, and no increase in execution failures pooled across seeds. No automatic extension.',
    limitations='Repeated validation is exploratory; results from two seeds reuse the same questions and are not independent test samples.',
)
LAUNCH = (broad_screen.LAUNCH
          .replace('broad-screen-1k-v1', NAME)
          .replace('homeworks.hw2.broad_screen', 'homeworks.hw2.batch_order_screen')
          .replace('timeout=2400', 'timeout=1200')
          .replace('SHORT_SCREEN_STARTED', 'BATCH_ORDER_STARTED')
          .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
                   'Hard bound 1200 seconds; four matched 1024-update arms only.'))


def select_pairs(results):
    pairs = {}
    failures = 0
    for seed in (42,1042):
        control, changed = results[f'control_s{seed}'], results[f'shuffled_s{seed}']
        gains = {stage:changed[stage]['numeric_correct']-control[stage]['numeric_correct']
                 for stage in ('pilot','confirmation')}
        delta = sum(changed[s]['prediction_failures']-control[s]['prediction_failures'] for s in gains)
        failures += delta
        pairs[str(seed)] = dict(split_gains=gains, total_gain=sum(gains.values()),
                               failure_delta=delta)
    confirmed = all(p['total_gain']>0 and min(p['split_gains'].values())>=0 for p in pairs.values()) and failures<=0
    return dict(results=results, paired_seeds=pairs, pooled_failure_delta=failures,
                confirmed=confirmed, training_updates_per_arm=1024,
                public_test_used_for_selection=False, long_run_started=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    if parser.parse_args().gpu:
        broad_screen.run(prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=select_pairs)
    else:
        broad_screen.prepare(prepared=PREPARED, run_dir=RUN, plan=PLAN, launcher=LAUNCH)
