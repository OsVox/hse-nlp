"""Three short matched arms testing focal loss on unchanged original tokens."""
import argparse
from copy import deepcopy

from . import broad_screen
from .short_pilots import ROOT

NAME='focal-screen-1k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PLAN=deepcopy(broad_screen.PLAN)
PLAN.update(final_step=71024,save_every=71024,eval_every=71024,
            training_updates_per_arm=1024,added_example_presentations_per_arm=131072,
            arms={'control':{},'focal1':{'focal_gamma':1.0},'focal2':{'focal_gamma':2.0}},
            overlap_cpu_scoring=True,outer_timeout_seconds=1200,
            hypothesis='Reduce easy-token influence with standard focal modulation (1-p)^gamma; original vocabulary, examples and programs remain unchanged.',
            limitations='Focal loss changes gradient magnitude as well as relative token emphasis. Same nominal LR is not identical effective step size. Reused held-out questions are exploratory.',
            selection='Require positive gain in both held-out subsets and no increase in total execution failures; repeat survivors with another seed before combination or extension. No automatic long run.',
            research='https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html; mechanism from detection, not evidence of benefit here')
LAUNCH=(broad_screen.LAUNCH.replace('broad-screen-1k-v1',NAME)
        .replace('homeworks.hw2.broad_screen','homeworks.hw2.focal_screen')
        .replace('timeout=2400','timeout=1200')
        .replace('SHORT_SCREEN_STARTED','FOCAL_SCREEN_STARTED')
        .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
                 'Hard bound 1200 seconds; three matched 1024-update arms only.'))


def select(results):
    result=broad_screen.select_candidates(results)
    result['shortlist']=[r['arm'] for r in result['ranking'] if r['eligible'] and min(r['split_gains'])>0]
    result['training_updates_per_arm']=1024
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
