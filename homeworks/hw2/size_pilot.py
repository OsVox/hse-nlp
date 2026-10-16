"""Bounded width/capacity screen using the replicated scratch Muon recipe."""
import argparse
from copy import deepcopy
from . import depth_pilot,tokenizer_pilot
from .short_pilots import ROOT

NAME='size-muon-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
CONFIGS={'small':dict(d_model=320,d_ff=960,num_heads=5),
         'control':dict(d_model=512,d_ff=1536,num_heads=8),
         'large':dict(d_model=704,d_ff=2112,num_heads=11)}
PLAN=deepcopy(depth_pilot.PLAN)
PLAN.update(seed=42,eval_every=512,outer_timeout_seconds=1400,training_timeout_seconds=600,
    arms={n:'original8192' for n in CONFIGS},
    arm_options={n:dict(optimizer='muon',learning_rate=.0003,muon_learning_rate=.001,**c)
                 for n,c in CONFIGS.items()},
    expected_parameters={'small':17580480,'control':42481152,'large':78146112},
    hypothesis='Compare early learning versus model capacity with the replicated Muon recipe, fixed 6/4 depth and 64-dimensional heads.',
    matched='Original tokenizer, data/targets, BF16, batch128, 2048 updates, warmup200, cosine schedule, and seed42. Width, feed-forward width and head count scale together. Shapes require different random tensors.',
    selection='At matched examples, require positive final held-out gains on both subsets and no extra total execution failures versus the fresh42M control. Record token-loss curves and actual GPU time separately. Repeat any survivor with another seed, then compare at matched GPU-time before longer training.',
    limitations='Equal updates do not match GPU cost. Shared rates are not size-specific tuning. Short token-loss curves and reused held-out scores do not establish compute-optimal size or mature accuracy. No automatic long run.',
    next_stages=['Replicate a surviving size with a second seed.','Compare survivors at matched measured GPU time.','Only then combine confirmed improvements in a longer scratch run.'])
LAUNCH=(depth_pilot.LAUNCH.replace(depth_pilot.NAME,NAME)
    .replace('homeworks.hw2.depth_pilot','homeworks.hw2.size_pilot')
    .replace('DEPTH_PILOT_STARTED','SIZE_PILOT_STARTED')
    .replace('three fresh depth-balance 2048-update arms only.','three fresh width/capacity 2048-update arms only.')
    .replace("    drive.mount('/content/drive', timeout_ms=180000)",
             "    assert 'RTX PRO 6000' in torch.cuda.get_device_name()\n    drive.mount('/content/drive', timeout_ms=180000)"))


def select(results):
    ranking=[]
    for name in ('small','large'):
        gains={s:results[name][s]['numeric_correct']-results['control'][s]['numeric_correct']
               for s in ('pilot','confirmation')}
        failures=sum(results[name][s]['prediction_failures']-results['control'][s]['prediction_failures'] for s in gains)
        ranking.append(dict(arm=name,split_gains=gains,failure_delta=failures,
                            eligible=min(gains.values())>0 and failures<=0))
    return dict(results=results,ranking=ranking,shortlist=[r['arm'] for r in ranking if r['eligible']],
        training_from_scratch=True,all_parameters_trained=True,training_updates_per_arm=2048,
        public_test_used_for_selection=False,long_run_started=False,equal_gpu_time=False,
        optimal_model_size_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
