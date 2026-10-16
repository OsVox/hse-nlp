"""Final bounded scratch comparison at matched measured training time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

from . import size_pilot
from .async_scoring import ScoringJobs
from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, copy_verified, restore_data
from .tokenizer_pilot import training_command

NAME='matched-curve-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
INPUTS=Path('/content/matched-curve-inputs')
CONFIGS={
    'adamw42':dict(d_model=512,d_ff=1536,num_heads=8,optimizer='adamw'),
    'muon42':dict(d_model=512,d_ff=1536,num_heads=8,optimizer='muon',muon_learning_rate=.001),
    'muon78':dict(d_model=704,d_ff=2112,num_heads=11,optimizer='muon',muon_learning_rate=.001),
}
OPTIONS=dict(steps=92456,batch_size=128,accumulation=1,seed=42,warmup=500,
    learning_rate=.0003,save_every=92456,eval_every=92456,
    max_training_seconds=600.,checkpoint_training_seconds=300.,
    precision='bf16',group_window=32,input_pipeline='encoded',prefetch_batches=2,
    encoder_layers=6,decoder_layers=4,initialization_vocab_size=16384,
    coverage_index=str(INPUTS/'index'),coverage_rows_per_shard=4096,
    untruncated_training=True,microbatch_token_budget=40960,microbatch_attention_budget=9961472,
    numeric_token_weight=1.,prediction_batch_size=64,
    prediction_max_input_tokens=0,prediction_max_new_tokens=1280)
PLAN=dict(options=OPTIONS,arms=CONFIGS,training_timeout_seconds=1050,outer_timeout_seconds=3900,
    hypothesis='Compare AdamW42 versus Muon42, and Muon42 versus Muon78, after a longer early-learning horizon at equal measured training time. The earlier 2048-step size screen did not establish the later learning curves.',
    training_budget='600 measured seconds per arm including data waits and optimizer updates, excluding checkpoint and evaluation work; report all overhead separately. Stop at a completed optimizer update. Save at300 and600 seconds.',
    selection='The final checkpoint chooses candidates: positive executed-answer gains on both fresh subsets with no extra total execution failures relative to AdamW42. If both Muon sizes pass, promote78 only if it also improves both subsets over42 without added failures; otherwise promote42. Repeat the provisional winner versus AdamW42 with a second seed before full training. If neither passes, retain AdamW42 for the full pass. Public scores are reporting only.',
    reference='All models start from random weights; same original tokenizer, seed, full-sequence stream, unweighted token loss, full-pass LR schedule.42M models have identical initialization. Size78 has a different shape, matched by measured time rather than updates.',
    schedule='Warmup500updates then cosine to10% of peak over the calculated92456-update full pass. Pilots do not restart a short decay cycle. Resume with optimizer, scheduler, random state and consumed-row cursor.',
    validation='2048 fixed original question groups balanced acrossshards1..59. Every copy is excluded from training, as are old validation groups. Shard0 question texts excluded to avoid tokenizer-question overlap; embedded program text was not audited. v1single-shard set retired before model evaluation.',
    decoding='Strict greedy with complete questions and max1280 generated tokens, above audited1208-token longest original target. Record generation-cap hits; no repairs, external data or teacher targets.',
    limitations='Different model sizes process different row counts in equal time. One short comparison and its repeat do not establish a universal optimum. Fresh holdout references may be unsupported or incorrect; report usable denominators and failures.',
    long_run_started=False,public_test_used_for_selection=False)
LAUNCH=(size_pilot.LAUNCH.replace(size_pilot.NAME,NAME)
    .replace('homeworks.hw2.size_pilot','homeworks.hw2.learning_curve')
    .replace('timeout=1400','timeout=3900')
    .replace('SIZE_PILOT_STARTED','MATCHED_CURVE_STARTED')
    .replace('Hard bound 1400 seconds; three fresh width/capacity 2048-update arms only.',
             'Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.'))


def comparison(candidate, control):
    gains={s:candidate[s]['numeric_correct']-control[s]['numeric_correct'] for s in ('pilot','confirmation')}
    failures=sum(candidate[s]['prediction_failures']-control[s]['prediction_failures'] for s in gains)
    return dict(gains=gains,failure_delta=failures,passes=min(gains.values())>0 and failures<=0)


def select(results):
    comparisons={n:comparison(results[n],results['adamw42']) for n in ('muon42','muon78')}
    size=comparison(results['muon78'],results['muon42'])
    chosen='adamw42'
    if comparisons['muon42']['passes']:chosen='muon42'
    if comparisons['muon78']['passes'] and (chosen=='adamw42' or size['passes']):chosen='muon78'
    return dict(results=results,comparisons=comparisons,size_comparison=size,provisional_winner=chosen,
                second_seed_required=chosen!='adamw42',public_test_used_for_selection=False,
                long_run_started=False,full_pass_completed=False)


def verify_checkpoint(checkpoint, step, eligible):
    marker=json.loads((checkpoint/'checkpoint.complete.json').read_text())
    for filename,digest in marker['files'].items():
        assert Path(filename).name==filename and sha256(checkpoint/filename)==digest
    coverage=json.loads((checkpoint/'coverage.json').read_text())
    r=coverage['report']
    assert coverage['optimizer_step']==step and r['start_step']==0
    assert r['presentations']==r['unique_rows_since_tracking_started'] and 0<r['presentations']<eligible
    assert r['repeated_presentations_since_tracking_started']==r['completed_full_passes']==0
    assert r['source_length'] is None and r['target_length'] is None
    assert r['untruncated_tokens']==coverage['state']['untruncated_tokens']
    provenance=json.loads((checkpoint/'predictions.provenance.json').read_text())
    assert provenance['num_beams']==1 and not provenance['do_sample'] and not provenance['syntax_retry_beams']
    assert provenance['max_input_tokens'] is None and provenance['max_new_tokens']==1280
    assert (checkpoint/'results.complete.json').exists()
    return r


def run(*, prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=select):
    PREPARED, RUN, PLAN, select = prepared, run_dir, plan, selection_fn
    CONFIGS, OPTIONS = PLAN['arms'], PLAN['options']
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory>=40*1024**3
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    if 'runtime_versions' in ready:
        from .runtime_guard import verify_runtime_versions
        verify_runtime_versions(ready['runtime_versions'])
    assert ready['cpu_verified'] and ready['plan']==PLAN and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    for filename,digest in ready['tokenizers']['original8192'].items():
        assert sha256(Path('pilot-tokenizers/original8192')/filename)==digest
    RUN.mkdir();(RUN/'runtime-manifest.json').write_text(json.dumps(ready,indent=2))
    started=time.monotonic();status=dict(completed=False,gpu=torch.cuda.get_device_name())
    scoring=ScoringJobs();scores={};curves={};initial={}
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            INPUTS.mkdir(exist_ok=False)
            archive=INPUTS/'fresh.tar.gz';copy_verified(Path(ready['fresh_holdout']['drive_path']),archive)
            assert sha256(archive)==ready['fresh_holdout']['sha256']
            with tarfile.open(archive) as z:z.extractall(INPUTS,filter='data')
            prep=ready['fresh_preparation']
            assert sha256(INPUTS/'questions.jsonl')==prep['questions_sha256']
            assert sha256(INPUTS/'index/manifest.json')==prep['index_manifest_sha256']
            for arm_name,config in CONFIGS.items():
                arm=RUN/arm_name;arm.mkdir()
                options=dict(OPTIONS,**config,tokenizer_dir=Path('pilot-tokenizers/original8192'),
                    cache_dir=CACHE,output_dir=Path('/content/matched-curve-training')/arm_name,drive_root=arm)
                command=training_command(options)
                (arm/'launch.json').write_text(json.dumps(dict(command=command),indent=2))
                with (arm/'training.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=PLAN['training_timeout_seconds'])
                budget=json.loads((arm/'compute-budget.json').read_text())
                assert budget['stop_reason']=='training_time_budget' and 600<=budget['training_seconds']<610
                assert len(budget['checkpoint_steps'])==2
                curves[arm_name]=[]
                for step in budget['checkpoint_steps']:
                    checkpoint=arm/f'step-{step:06d}'
                    coverage=verify_checkpoint(checkpoint,step,prep['new_eligible_rows'])
                    timing=json.loads((checkpoint/'training-budget.json').read_text())
                    assert timing['optimizer_step']==step
                    initial[arm_name]=json.loads((checkpoint/'run_config.json').read_text())['initial_model_sha256']
                    key=f'{arm_name}:{step}';scores[key]={}
                    curves[arm_name].append(dict(step=step,coverage=coverage,timing=timing,key=key))
                    for stage,offset in [('pilot',0),('confirmation',1024)]:
                        output=checkpoint/(stage+'.json')
                        command=[sys.executable,'-m','homeworks.hw2.numeric_eval','--model-dir',str(checkpoint),
                            '--cache-dir',str(CACHE),'--examples-file',str(INPUTS/'questions.jsonl'),
                            '--output',str(output),'--offset',str(offset),'--rows','1024','--beams','1',
                            '--max-input-tokens','0','--max-new-tokens','1280']
                        with output.with_suffix('.log').open('w') as log:
                            subprocess.run(command+['--generate-only'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
                        scoring.start(key,stage,command+['--score-only'],output)
                        scoring.collect(scores)
                if {'adamw42','muon42'} <= initial.keys():assert initial['adamw42']==initial['muon42']
                (RUN/'progress.json').write_text(json.dumps(dict(curves=curves,scores=scores),indent=2))
            scoring.collect(scores,wait=True)
            for points in curves.values():
                for point in points:point['results']=scores[point.pop('key')]
            final={name:points[-1]['results'] for name,points in curves.items()}
            (RUN/'selection.json').write_text(json.dumps(select(final),indent=2))
            (RUN/'learning-curves.json').write_text(json.dumps(curves,indent=2))
            (RUN/'initialization.json').write_text(json.dumps(initial,indent=2))
            status['completed']=True
        finally:
            scoring.close();monitor.terminate();monitor.wait(timeout=10)
            (RUN/'scoring-execution.json').write_text(json.dumps(scoring.completed,indent=2))
            status['elapsed_seconds']=time.monotonic()-started
            (RUN/'execution.json').write_text(json.dumps(status,indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.log'}:z.write(path,path.relative_to(RUN))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args();run()
