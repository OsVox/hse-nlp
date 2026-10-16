"""Short optimizer-transfer test from our own verified scratch-trained model."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from . import broad_screen
from .short_pilots import ROOT

NAME = 'muon-transfer-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PLAN = deepcopy(broad_screen.PLAN)
PLAN.update(parent_step=73024, final_step=74048, save_every=74048, eval_every=74048,
    seed=42, numeric_token_weight=2., overlap_cpu_scoring=True,
    added_example_presentations_per_arm=131072,
    training_timeout_seconds=360, outer_timeout_seconds=1200,
    parent_model_sha256='cad272fa2644c3876a577affc6f8559be9d5efce4e0f5c66f8f3ea6b7fb84cb6',
    arms={
        'control': {'numeric_token_weight':2.},
        'adamw_reset': {'numeric_token_weight':2., 'reset_optimizer':True},
        'muon_low': {'numeric_token_weight':2., 'reset_optimizer':True,
                     'optimizer':'muon','muon_learning_rate':.0001},
        'muon_high': {'numeric_token_weight':2., 'reset_optimizer':True,
                      'optimizer':'muon','muon_learning_rate':.0003},
    },
    hypothesis='Do replicated fresh-model Muon gains transfer to the mature AdamW-trained checkpoint?',
    matched='Identical weights, original tokenizer/data, data/dropout seed, 1024 updates, batch128, and numeric loss weight2. AdamW rate 3e-5 in every arm. Fresh-AdamW control isolates resetting moments from optimizer choice.',
    selection='Compare Muon with the stronger of continued and reset AdamW using summed held-out agreements, then fewer failures. Require improvement on both subsets and no extra total failures, plus no loss against the parent on either subset. Repeat any survivor with another data/dropout seed before extension. Public scores do not select recipes.',
    limitations='Small rate screen; reused held-out sets. Equal updates do not match GPU time. A transfer result does not establish optimal fresh-model size. No automatic long run.',
    expansion='Save each checkpoint and exact greedy outputs in Drive, then export and submit every checkpoint to the bot.')
LAUNCH = (broad_screen.LAUNCH.replace(broad_screen.NAME,NAME)
    .replace('homeworks.hw2.broad_screen','homeworks.hw2.muon_transfer')
    .replace('timeout=2400','timeout=1200')
    .replace('SHORT_SCREEN_STARTED','MUON_TRANSFER_STARTED')
    .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
             'Hard bound 1200 seconds; four matched 1024-update arms only.')
    .replace('assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()',
             "assert hasattr(torch.optim, 'Muon')\n    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n    assert 'RTX PRO 6000' in torch.cuda.get_device_name()"))


def select(results):
    def key(name):
        s=results[name]
        return (-sum(v['numeric_correct'] for v in s.values()),
                sum(v['prediction_failures'] for v in s.values()),name)
    reference=min(['control','adamw_reset'],key=key)
    parent=json.loads((PREPARED/'ready.json').read_text())['parent_heldout']
    ranking=[]
    for name in ['muon_low','muon_high']:
        gains={stage:results[name][stage]['numeric_correct']-results[reference][stage]['numeric_correct']
               for stage in ('pilot','confirmation')}
        parent_gains={stage:results[name][stage]['numeric_correct']-parent[stage]['numeric_correct']
                      for stage in gains}
        failures=sum(results[name][stage]['prediction_failures']-results[reference][stage]['prediction_failures']
                     for stage in gains)
        ranking.append(dict(arm=name,split_gains=gains,parent_gains=parent_gains,
            failure_delta=failures,eligible=min(gains.values())>0 and min(parent_gains.values())>=0 and failures<=0))
    return dict(results=results,adamw_reference=reference,parent_heldout=parent,ranking=ranking,
        shortlist=[r['arm'] for r in ranking if r['eligible']],training_updates_per_arm=1024,
        all_parameters_trained=True,own_scratch_checkpoint=True,
        public_test_used_for_selection=False,long_run_started=False,second_seed_required=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
