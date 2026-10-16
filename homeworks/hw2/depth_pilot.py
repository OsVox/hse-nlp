"""Short matched-parameter encoder/decoder balance screen using replicated Muon."""
import argparse
from copy import deepcopy
from . import optimizer_pilot, tokenizer_pilot
from .short_pilots import ROOT

NAME='depth-muon-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PLAN=deepcopy(optimizer_pilot.PLAN)
CONFIGS={
    'control':dict(encoder_layers=6,decoder_layers=4,d_ff=1536),
    'encoder_heavy':dict(encoder_layers=8,decoder_layers=2,d_ff=1664),
    'decoder_heavy':dict(encoder_layers=4,decoder_layers=6,d_ff=1408)}
PLAN.update(seed=42,outer_timeout_seconds=1400,
    arms={name:'original8192' for name in CONFIGS},
    arm_options={name:dict(optimizer='muon',learning_rate=.0003,
                          muon_learning_rate=.001,**config)
                 for name,config in CONFIGS.items()},
    expected_parameters={'control':42481152,'encoder_heavy':42349056,'decoder_heavy':42613248},
    require_identical_initialization=False,
    hypothesis='Does allocating more layers to the encoder or decoder improve executed-answer learning at nearly equal parameter count?',
    matched='All models train from scratch with Muon 0.001 plus embedding/vector AdamW 0.0003, seed 42, original tokenizer/data/targets, batch 128 and 2048 updates. Width is 512 and head count 8. Feed-forward width adjusts to keep parameters within 0.32% of baseline. Different shapes cannot have identical random weights; record each initialization hash.',
    selection='Require positive gains on both held-out subsets and no added total execution failures versus the freshly trained Muon 6/4 control. Repeat any survivor with seed 1042 before combining or extending. Public scores are logged only.',
    limitations='This tests depth balance with a small compensating feed-forward-width change. Equal parameters and examples do not equal GPU time. Architecture-specific rate tuning is not covered; results only apply to this shared recipe. Early learning and reused validation are exploratory.',
    next_stages=['Repeat surviving depth recipe versus Muon control with another seed.',
                 'Compare learning curves and measured GPU time, then test model size.',
                 'Only combine replicated improvements into a longer run.'])
LAUNCH=(optimizer_pilot.LAUNCH.replace(optimizer_pilot.NAME,NAME)
    .replace('homeworks.hw2.optimizer_pilot','homeworks.hw2.depth_pilot')
    .replace('timeout=1800','timeout=1400')
    .replace('OPTIMIZER_PILOT_STARTED','DEPTH_PILOT_STARTED')
    .replace('Hard bound 1800 seconds; four fresh full-model 2048-update arms only.',
             'Hard bound 1400 seconds; three fresh depth-balance 2048-update arms only.'))


def select(results):
    ranking=[]
    for name in ('encoder_heavy','decoder_heavy'):
        gains=[results[name][s]['numeric_correct']-results['control'][s]['numeric_correct']
               for s in ('pilot','confirmation')]
        failures=sum(results[name][s]['prediction_failures']-results['control'][s]['prediction_failures']
                     for s in ('pilot','confirmation'))
        ranking.append(dict(arm=name,split_gains=gains,failure_delta=failures,
                            eligible=min(gains)>0 and failures<=0))
    return dict(results=results,ranking=ranking,
                shortlist=[r['arm'] for r in ranking if r['eligible']],
                training_from_scratch=True,all_parameters_trained=True,
                training_updates_per_arm=2048,public_test_used_for_selection=False,
                long_run_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
