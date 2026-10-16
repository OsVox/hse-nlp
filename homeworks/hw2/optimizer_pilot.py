"""Fresh full-model AdamW/Muon screen with separate learning-rate choices."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from . import tokenizer_pilot
from .short_pilots import ROOT

NAME='optimizer-scratch-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PLAN=deepcopy(tokenizer_pilot.PLAN)
PLAN.update(seed=42,eval_every=2048,training_timeout_seconds=480,
    outer_timeout_seconds=1800,
    throughput_guard_estimated_train_seconds=400,
    arms={name:'original8192' for name in ['control','adamw_high','muon_low','muon_high']},
    arm_options={
        'control':{'optimizer':'adamw','learning_rate':.0003},
        'adamw_high':{'optimizer':'adamw','learning_rate':.001},
        'muon_low':{'optimizer':'muon','learning_rate':.0003,'muon_learning_rate':.001},
        'muon_high':{'optimizer':'muon','learning_rate':.0003,'muon_learning_rate':.003}},
    hypothesis='Does native Muon learn executed-answer accuracy faster from scratch than AdamW after a small optimizer-specific learning-rate sweep?',
    matched='Identical full random weights, tokenizer, original data, dropout seed, batch size, update count and target loss. All model parameters train. Muon applies only to hidden matrices; embeddings and vectors use AdamW. No frozen backbone.',
    selection='Choose the better AdamW arm by summed held-out numeric agreements, breaking ties by fewer failures then lower rate. Muon must beat that arm on BOTH subsets with no extra total failures. Repeat best recipes with a second seed before combining or extending.',
    limitations='Two-rate screens are not exhaustive tuning. Equal updates match example count, not GPU cost. Log actual examples, tokens, process time including setup, driver wall time and GPU telemetry. A fresh-model result does not establish optimizer-switch benefits on an AdamW-trained checkpoint. Public scores never select a recipe.',
    provenance='Original supplied examples and random weights only. Same original 8k tokenizer in every arm. Native PyTorch Muon with match_rms_adamw scaling; no imported training data or pretrained model.',
    reference='https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html',
    next_stages=['Repeat best AdamW and Muon recipes with a second seed.',
        'Only after replication, test full-model continuation or a longer fresh run with matched controls.',
        'Test encoder/decoder balance, then model size with measured equal GPU-time budgets.'])
LAUNCH=(tokenizer_pilot.LAUNCH.replace(tokenizer_pilot.NAME,NAME)
    .replace('homeworks.hw2.tokenizer_pilot','homeworks.hw2.optimizer_pilot')
    .replace('timeout=1600','timeout=1800')
    .replace('TOKENIZER_PILOT_STARTED','OPTIMIZER_PILOT_STARTED')
    .replace('Hard bound 1600 seconds; four scratch 2048-update arms only.',
             'Hard bound 1800 seconds; four fresh full-model 2048-update arms only.')
    .replace('assert torch.cuda.is_available()',
             "assert hasattr(torch.optim, 'Muon'), 'Prepare torch >= 2.9 on CPU before GPU allocation'\n    assert torch.cuda.is_available()"))


def select(results):
    def quality(name):
        stages=results[name].values()
        return (-sum(s['numeric_correct'] for s in stages),
                sum(s['prediction_failures'] for s in results[name].values()),
                PLAN['arm_options'][name]['learning_rate'])
    reference=min(['control','adamw_high'],key=quality)
    ranking=[]
    for name in ['muon_low','muon_high']:
        gains=[results[name][s]['numeric_correct']-results[reference][s]['numeric_correct']
               for s in ['pilot','confirmation']]
        failures=sum(results[name][s]['prediction_failures']-results[reference][s]['prediction_failures']
                     for s in ['pilot','confirmation'])
        ranking.append(dict(arm=name,split_gains=gains,failure_delta=failures,
                            eligible=min(gains)>0 and failures<=0))
    return dict(results=results,adamw_reference=reference,ranking=ranking,
        shortlist=[r['arm'] for r in ranking if r['eligible']],
        training_from_scratch=True,all_parameters_trained=True,
        training_updates_per_arm=2048,public_test_used_for_selection=False,
        long_run_started=False)


def throughput_preflight(run_dir):
    """Time discarded full-size GPU steps before committing to four arms."""
    import gc
    import time
    import torch
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast
    from .train import new_model, encode_batch
    from .optimizers import make_optimizer
    from .short_pilots import CACHE
    tokenizer=PreTrainedTokenizerFast.from_pretrained('pilot-tokenizers/original8192')
    rows=pq.read_table(CACHE/'part-00000.parquet',columns=['question','code']).to_pylist()
    excluded={r['question'] for r in rows[:1024]}
    examples=[r for r in rows[1024:] if r['question'] not in excluded][:128]
    batch=encode_batch(examples,tokenizer,torch.device('cuda'))
    results={}
    for name in ['control','muon_low']:
        torch.manual_seed(42)
        model=new_model(tokenizer,**{k:PLAN[k] for k in ['d_model','d_ff',
            'encoder_layers','decoder_layers','num_heads','initialization_vocab_size']}).cuda().train()
        args=PLAN['arm_options'][name]
        optimizer=make_optimizer(model,args['optimizer'],learning_rate=args['learning_rate'],
                                  muon_learning_rate=args.get('muon_learning_rate'))
        times=[]
        for step in range(11):
            torch.cuda.synchronize();start=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**batch,use_cache=False).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            assert torch.isfinite(loss)
            if step>=3:times.append(elapsed)
        seconds=sum(times)/len(times)
        results[name]=dict(seconds_per_step=seconds,
            estimated_training_seconds=seconds*PLAN['steps'],
            input_shape=list(batch['input_ids'].shape),target_shape=list(batch['labels'].shape))
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
    record=dict(results=results,torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),updates_per_optimizer=11,
        scope='Discarded timing probe on original train examples; no checkpoint or accuracy result. Fixed-batch throughput is an estimate, not full-run time.')
    (run_dir/'throughput-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print('OPTIMIZER_THROUGHPUT',json.dumps(record),flush=True)
    assert max(v['estimated_training_seconds'] for v in results.values()) <= PLAN['throughput_guard_estimated_train_seconds'], 'Stop: optimizer throughput exceeds this cheap-pilot budget'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args()
    import torch
    assert hasattr(torch.optim,'Muon')
    tokenizer_pilot.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select,
                        runtime_preflight_fn=throughput_preflight)
