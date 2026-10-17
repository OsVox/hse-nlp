"""Scale the selected AdamW recipe with bounded, resumable validation intervals."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

from . import adamw_size_confirm, adamw_size_curve, learning_curve
from .async_scoring import ScoringJobs
from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, copy_verified, restore_data
from .tokenizer_pilot import training_command

NAME = 'selected-scale-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
INPUTS = Path('/content/selected-scale-inputs')
PASS_UPDATES = 92456


def selected_recipe(prior, confirmation=None):
    checked = adamw_size_confirm.first_seed(prior)
    if not checked['second_seed_required']:
        return 'adamw42'
    if confirmation is None:
        raise ValueError('AdamW78 requires paired second-seed confirmation')
    result = adamw_size_confirm.paired_selection(confirmation['results'], prior)
    if result['full_pass_recipe'] != confirmation['full_pass_recipe']:
        raise ValueError('Saved confirmation disagrees with its measurements')
    return result['full_pass_recipe']


def build_plan(prior, confirmation, config, parent_step, seconds_per_update):
    recipe = selected_recipe(prior, confirmation)
    width = 704 if recipe == 'adamw78' else 512
    # Resume CLI architecture flags are deliberately null; the saved model
    # configuration is authoritative for those fields.
    actual = dict(config)
    for key, model_key in dict(d_model='d_model',d_ff='d_ff',num_heads='num_heads',
                              encoder_layers='num_layers',decoder_layers='num_decoder_layers').items():
        if actual.get(key) is None:
            actual[key] = config.get('model_config',{}).get(model_key)
    required = dict(optimizer='adamw',d_model=width,seed=42,untruncated_training=True,
        d_ff=width*3,num_heads=width//64,
        batch_size=128,accumulation=1,encoder_layers=6,decoder_layers=4,
        group_window=32,input_pipeline='encoded',prefetch_batches=2,
        microbatch_token_budget=40960,microbatch_attention_budget=9961472,
        learning_rate=.0003,warmup=500,steps=PASS_UPDATES)
    for key, value in required.items():
        if actual.get(key) != value:
            raise ValueError(f'Unexpected parent setting: {key}')
    if not 0 < seconds_per_update < 2:
        raise ValueError('Invalid measured training speed')
    if recipe == 'adamw42' and parent_step != PASS_UPDATES:
        raise ValueError('Continue the completed first-pass reference')
    if recipe == 'adamw78' and not 0 < parent_step < PASS_UPDATES:
        raise ValueError('Continue the first-seed AdamW78 pilot')
    final_step = PASS_UPDATES * (2 if recipe == 'adamw42' else 1)
    options = deepcopy(learning_curve.OPTIONS)
    options.update(optimizer='adamw',steps=final_step,save_every=final_step,
        eval_every=final_step,max_training_seconds=1000.,checkpoint_training_seconds=None,
        coverage_index=str(INPUTS/'index'))
    for key in ('initialization_vocab_size','encoder_layers','decoder_layers'):
        options.pop(key)
    # At the completed 42M pass, continue from its final 3e-5 LR without a jump.
    # Optimizer moments and stream state survive; only the explicit LR phase changes.
    if recipe == 'adamw42':
        options.update(learning_rate=.00003,warmup=0)
    expected = (final_step-parent_step)*seconds_per_update
    total_budget = math.ceil(expected*1.35)
    max_segments = math.ceil(total_budget/1000)+1
    return dict(recipe=recipe,parent_step=parent_step,final_step=final_step,
        final_completed_passes=final_step//PASS_UPDATES,options=options,
        restart_first_schedule=recipe=='adamw42',patience=3,
        measured_seconds_per_update=seconds_per_update,
        estimated_training_seconds=expected,total_training_budget_seconds=total_budget,
        max_segments=max_segments,segment_timeout_seconds=1450,
        outer_timeout_seconds=total_budget+max_segments*500+600,
        selection='Strict held-out gains in both subsets with no added total execution failures versus the best qualifying checkpoint. Stop after three consecutive intervals without such a gain.',
        coverage='Resume the exact consumed-row cursor. Stop at the planned complete pass, the measured training budget, or validation patience; report partial coverage honestly.',
        schedule='78M preserves its original full-pass schedule. 42M starts a second-pass cosine phase at the first pass final LR3e-5, decaying to3e-6, without resetting optimizer moments.',
        public_test_used_for_selection=False,additional_sweep_authorized=False)


def validation_update(state, candidate, *, patience):
    # Reuse the comparison identity checks before treating a score as an improvement.
    result = adamw_size_curve.select({'adamw78': candidate},
        dict(results=state['best'],model_sha256=state['best_model_sha256']))
    improved = result['comparison']['passes']
    result_state = deepcopy(state)
    result_state['stale_intervals'] = 0 if improved else state['stale_intervals']+1
    if improved:
        result_state['best'] = candidate
        result_state['best_model_sha256'] = candidate['pilot']['model_sha256']
    return result_state, dict(improved=improved,comparison=result['comparison'],
        stop=result_state['stale_intervals']>=patience)


def launcher(plan):
    return (learning_curve.LAUNCH.replace(learning_curve.NAME,NAME)
        .replace('homeworks.hw2.learning_curve','homeworks.hw2.scaled_training')
        .replace('timeout=3900',f"timeout={plan['outer_timeout_seconds']}")
        .replace('MATCHED_CURVE_STARTED','SELECTED_SCALE_STARTED')
        .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
                 f"Hard bound{plan['outer_timeout_seconds']}seconds; selected recipe with validation stopping."))


def evaluate(checkpoint, examples, scoring):
    scores={'checkpoint':{}}
    for stage,offset in [('pilot',0),('confirmation',1024)]:
        output=checkpoint/(stage+'.json')
        command=[sys.executable,'-m','homeworks.hw2.numeric_eval',
            '--model-dir',str(checkpoint),'--cache-dir',str(CACHE),
            '--examples-file',str(examples),'--output',str(output),
            '--offset',str(offset),'--rows','1024','--beams','1',
            '--max-input-tokens','0','--max-new-tokens','1280']
        with output.with_suffix('.log').open('w') as log:
            subprocess.run(command+['--generate-only'],stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=180)
        scoring.start('checkpoint',stage,command+['--score-only'],output)
    scoring.collect(scores,wait=True)
    return scores['checkpoint']


def run():
    import torch
    from .runtime_guard import verify_runtime_versions
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text());plan=ready['plan']
    verify_runtime_versions(ready['runtime_versions'])
    assert ready['cpu_verified'] and ready['selection_verified'] and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    RUN.mkdir();copy_verified(PREPARED/'ready.json',RUN/'runtime-manifest.json')
    started=time.monotonic();status=dict(completed=False,gpu=torch.cuda.get_device_name())
    records=[];training_seconds=0.;scoring=ScoringJobs()
    state=dict(best=ready['parent_heldout'],best_model_sha256=ready['parent_model_sha256'],stale_intervals=0)
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset']);INPUTS.mkdir(exist_ok=False)
            for name,source,digest in [('fresh',Path(ready['fresh_holdout']['drive_path']),ready['fresh_holdout']['sha256']),
                                       ('parent',PREPARED/'parent.tar',ready['parent_archive_sha256'])]:
                archive=INPUTS/(name+'.tar');copy_verified(source,archive)
                assert sha256(archive)==digest
                with tarfile.open(archive) as z:z.extractall(INPUTS,filter='data')
            parent=INPUTS/'parent'
            assert sha256(parent/'model.safetensors')==ready['parent_model_sha256']
            assert build_plan(ready['size_selection'],ready.get('size_confirmation'),
                json.loads((parent/'run_config.json').read_text()),plan['parent_step'],
                plan['measured_seconds_per_update'])==plan
            assert all(value['model_sha256']==ready['parent_model_sha256']
                       for value in ready['parent_heldout'].values())
            assert sha256(INPUTS/'questions.jsonl')==ready['fresh_preparation']['questions_sha256']
            assert sha256(INPUTS/'index/manifest.json')==ready['fresh_preparation']['index_manifest_sha256']
            previous=json.loads((parent/'coverage.json').read_text())
            step=plan['parent_step']
            assert previous['optimizer_step']==step
            stop_reason='segment_limit'
            for index in range(plan['max_segments']):
                remaining=plan['total_training_budget_seconds']-training_seconds
                if remaining<1:
                    stop_reason='training_time_budget';break
                segment=RUN/f'segment-{index+1:02d}';segment.mkdir()
                options=dict(plan['options'],max_training_seconds=min(1000.,remaining),
                    resume_from=parent,cache_dir=CACHE,output_dir=Path('/content/scaled-training')/segment.name,
                    drive_root=segment,restart_schedule=index==0 and plan['restart_first_schedule'])
                command=training_command(options)
                (segment/'launch.json').write_text(json.dumps(dict(command=command),indent=2))
                with (segment/'training.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                                   timeout=plan['segment_timeout_seconds'])
                budget=json.loads((segment/'compute-budget.json').read_text())
                next_step=budget['final_step'];assert step<next_step<=plan['final_step']
                assert budget['checkpoint_steps']==[next_step]
                checkpoint=segment/f'step-{next_step:06d}'
                marker=json.loads((checkpoint/'checkpoint.complete.json').read_text())
                for name,digest in marker['files'].items():
                    assert Path(name).name==name and sha256(checkpoint/name)==digest
                assert (checkpoint/'results.complete.json').exists()
                coverage=json.loads((checkpoint/'coverage.json').read_text())
                assert coverage['optimizer_step']==next_step
                assert coverage['report']['presentations']-previous['report']['presentations']==budget['examples_processed']
                assert coverage['report']['source_length'] is None and coverage['report']['target_length'] is None
                training_seconds+=budget['training_seconds'];step=next_step
                values=evaluate(checkpoint,INPUTS/'questions.jsonl',scoring)
                state,decision=validation_update(state,values,patience=plan['patience'])
                records.append(dict(segment=segment.name,step=step,budget=budget,heldout=values,decision=decision))
                (RUN/'progress.json').write_text(json.dumps(dict(records=records,state=state,
                    training_seconds=training_seconds,step=step),indent=2))
                if step==plan['final_step']:
                    assert coverage['state']['cursor']==[plan['final_completed_passes'],0,0]
                    assert coverage['report']['presentations']==ready['fresh_preparation']['new_eligible_rows']*plan['final_completed_passes']
                    stop_reason='planned_pass_complete';break
                if decision['stop']:
                    stop_reason='validation_patience';break
                parent=checkpoint;previous=coverage
            status.update(completed=True,stop_reason=stop_reason,final_step=step,
                training_seconds=training_seconds,planned_pass_completed=step==plan['final_step'],
                best_model_sha256=state['best_model_sha256'])
        finally:
            scoring.close();monitor.terminate();monitor.wait(timeout=10)
            status['elapsed_seconds']=time.monotonic()-started
            (RUN/'execution.json').write_text(json.dumps(status,indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(RUN.rglob('*')):
                    if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.log'}:z.write(p,p.relative_to(RUN))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args();run()
