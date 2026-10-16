"""Short matched target-cleanup continuation; no new examples or solutions."""
import argparse
import json
from pathlib import Path
from . import broad_screen
from .short_pilots import ROOT

NAME = 'comment-continuation-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PLAN = json.loads((Path(__file__).parent/'experiments'/NAME/'plan.json').read_text())
LAUNCH = (broad_screen.LAUNCH.replace('broad-screen-1k-v1', NAME)
    .replace('homeworks.hw2.broad_screen', 'homeworks.hw2.comment_continuation')
    .replace('timeout=2400', 'timeout=900')
    .replace('SHORT_SCREEN_STARTED', 'COMMENT_CONTINUATION_STARTED')
    .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
             'Hard bound 900 seconds; two matched 1024-update arms only.'))


def select(results):
    result = broad_screen.select_candidates(results)
    for row in result['ranking']:
        row['eligible'] = row['eligible'] and min(row['split_gains']) > 0
    result['shortlist'] = [row['arm'] for row in result['ranking'] if row['eligible']]
    result['training_updates_per_arm'] = 1024
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', required=True)
    parser.parse_args()
    broad_screen.run(prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=select)
