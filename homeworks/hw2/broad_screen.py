"""Broad matched screening: eight hypotheses, 1,000 updates per arm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, copy_verified, restore_data
from .async_scoring import ScoringJobs

NAME = 'broad-screen-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PARENT = ROOT/'runs/full-42m-100k-v1/step-070000'
MODEL_HASH = '724b4c72914828f6f3facf72c1e7185953cd257df0cb29597f4b3bca5f24c0ac'
ARMS = {
    'control': {},
    'numeric2': {'numeric_token_weight': 2.0},
    'smooth005': {'label_smoothing': .05},
    'equal_question': {'example_balanced_loss': True},
    'dropout0': {'continuation_dropout': 0.0},
    'dropout005': {'continuation_dropout': .05},
    'lr6e5': {'learning_rate': .00006},
    'complete_targets': {'skip_truncated_targets': True},
}
PLAN = dict(parent_step=70000, final_step=71000, batch_size=128,
            learning_rate=.00003, warmup=50, save_every=1000, eval_every=1000,
            precision='bf16', group_window=32, mixed_rows_per_shard=4096,
            input_pipeline='encoded', prefetch_batches=2, prediction_batch_size=256,
            arms=ARMS, added_example_presentations_per_arm=128000,
            training_timeout_seconds=400, outer_timeout_seconds=2400,
            hypothesis='One factor per arm from identical 70k weights and optimizer; fixed seed and budget.',
            limitations='Filtered arm necessarily sees different examples, but full batches and update count match. Validation has been reused; effects are exploratory.',
            selection='Rank all arms on numeric agreements in held-out rows 256:1024. Shortlist at most three with positive total gain, no negative gain in either half, and no increase in total execution failures. Repeat survivors against control with a second seed; test a combination only after that. No automatic long training.',
            expansion='Persist every checkpoint and strict greedy test output to Drive; export and evaluate every checkpoint in bot. Public feedback is recorded separately.')

LAUNCH = '''from google.colab import drive, runtime
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3
    drive.mount('/content/drive', timeout_ms=180000)
    import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
    from pathlib import Path
    prepared=Path('/content/drive/MyDrive/HSE_NLP_HW2/prepared/broad-screen-1k-v1')
    ready=json.loads((prepared/'ready.json').read_text())
    assert ready['cpu_verified']
    work=Path('/content/broad-screen-source');work.mkdir(exist_ok=False)
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as z:z.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],check=True,timeout=180)
    sys.path.insert(0,str(work))
    from homeworks.hw2.runtime_guard import finalize_run
    local_log=work/'execution.log'
    full_log=local_log.open('w')
    full_process=subprocess.Popen([sys.executable,'-u','-m','homeworks.hw2.broad_screen','--gpu'],cwd=work,stdout=full_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish_full(full_process=full_process, full_log=full_log, prepared=prepared):
        status={'pid':full_process.pid,'started_utc':time.time()}
        try:
            await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=2400)
            status['returncode']=full_process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if full_process.poll() is None:
                os.killpg(full_process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=10)
                except asyncio.TimeoutError:os.killpg(full_process.pid,signal.SIGKILL)
        finally:
            finalize_run(full_log,status,prepared,local_log,runtime.unassign)
    full_task=asyncio.create_task(finish_full())
    print('SHORT_SCREEN_STARTED',full_process.pid,'Hard bound 2400 seconds; eight matched 1k arms only.')
except BaseException:
    runtime.unassign()
    raise
'''


def select_candidates(results):
    rows=[]
    for name, stages in results.items():
        gains=[stages[stage]['numeric_correct']-results['control'][stage]['numeric_correct']
               for stage in ['pilot','confirmation']]
        failures=sum(stages[stage]['prediction_failures']-results['control'][stage]['prediction_failures']
                     for stage in ['pilot','confirmation'])
        rows.append(dict(arm=name,gain=sum(gains),split_gains=gains,failure_delta=failures,
                         eligible=name!='control' and sum(gains)>0 and min(gains)>=0 and failures<=0))
    rows.sort(key=lambda row:(-row['gain'],row['failure_delta'],row['arm']))
    return dict(results=results,ranking=rows,shortlist=[r['arm'] for r in rows if r['eligible']][:3],
                training_updates_per_arm=1000,public_test_used_for_selection=False,
                long_run_authorized_by_evidence=False)


def prepare(*, prepared=PREPARED, run_dir=RUN, plan=PLAN, launcher=LAUNCH, preflight_fn=None):
    PREPARED, RUN, PLAN, LAUNCH = prepared, run_dir, plan, launcher
    import torch
    from transformers import T5ForConditionalGeneration
    assert not Path('/dev/nvidia0').exists() and not torch.cuda.is_available()
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists(), 'Inspect prior preparation before retrying'
    assert (PARENT.parent/'intentional-stop.json').exists()
    dataset = json.loads((ROOT/'prepared/ready.json').read_text())
    restore_data(dataset)
    local = Path('/content/broad-screen-preparation');local.mkdir(exist_ok=False)
    parent = local/'parent';parent.mkdir()
    manifest = json.loads((PARENT/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==MODEL_HASH
    for name,digest in manifest['files'].items():
        assert Path(name).name==name and sha256(PARENT/name)==digest
        copy_verified(PARENT/name,parent/name)
    copy_verified(PARENT/'checkpoint.complete.json',parent/'checkpoint.complete.json')
    torch.set_num_threads(2)
    model = T5ForConditionalGeneration.from_pretrained(parent,local_files_only=True)
    assert sum(p.numel() for p in model.parameters())==42481152
    optimizer = torch.optim.AdamW(model.parameters(),lr=PLAN['learning_rate'])
    state = torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['step']==PLAN['parent_step']
    optimizer.load_state_dict(state['optimizer'])
    for parameter,values in optimizer.state.items():
        assert values['exp_avg'].shape==parameter.shape==values['exp_avg_sq'].shape
        assert torch.isfinite(values['exp_avg']).all() and torch.isfinite(values['exp_avg_sq']).all()
    del state,optimizer,model
    extra_preflight = preflight_fn(parent, CACHE) if preflight_fn else None
    commit = Path('source_commit.txt').read_text().strip()
    PREPARED.mkdir(parents=True)
    with tarfile.open(local/'parent.tar','w') as z:z.add(parent,arcname='parent')
    copy_verified(local/'parent.tar',PREPARED/'parent.tar')
    with tarfile.open(local/'source.tar','w') as z:
        z.add('source_commit.txt')
        for path in sorted(Path('homeworks/hw2').glob('*.py')):z.add(path)
    copy_verified(local/'source.tar',PREPARED/'source.tar')
    compile(LAUNCH,'launch.py','exec')
    (PREPARED/'launch.py').write_text(LAUNCH)
    ready = dict(cpu_verified=True,source_commit=commit,plan=PLAN,dataset=dataset,
                 parent_model_sha256=MODEL_HASH,parent_sha256=sha256(PREPARED/'parent.tar'),
                 source_sha256=sha256(PREPARED/'source.tar'),launcher_sha256=sha256(PREPARED/'launch.py'))
    if extra_preflight is not None:
        (PREPARED/'tokenization-audit.json').write_text(json.dumps(extra_preflight,indent=2))
        ready['extra_preflight'] = extra_preflight
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2))
    assert json.loads((PREPARED/'ready.json').read_text())==ready
    print('SCREEN_CPU_READY',json.dumps(ready),flush=True)


def run(*, prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=None):
    PREPARED, RUN, PLAN = prepared, run_dir, plan
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory>=40*1024**3
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['plan']==PLAN and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    RUN.mkdir(parents=True);copy_verified(PREPARED/'ready.json',RUN/'plan.json')
    began=time.monotonic();status=dict(completed=False,gpu=torch.cuda.get_device_name())
    scoring = ScoringJobs()
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            local=Path('/content/broad-screen-inputs');local.mkdir(exist_ok=False)
            parent_archive = Path(ready.get('parent_archive_path',str(PREPARED/'parent.tar')))
            copy_verified(parent_archive,local/'parent.tar')
            assert sha256(local/'parent.tar')==ready['parent_sha256']
            with tarfile.open(local/'parent.tar') as z:z.extractall(local,filter='data')
            results = {}
            for name, changes in PLAN['arms'].items():
                arm = RUN/name; arm.mkdir()
                options={k:PLAN[k] for k in ['batch_size','learning_rate','warmup','save_every','eval_every','precision','group_window','mixed_rows_per_shard','input_pipeline','prefetch_batches','prediction_batch_size']}
                options.update(resume_from=local/'parent',steps=PLAN['final_step'],accumulation=1,seed=42,cache_dir=CACHE,output_dir=Path('/content/broad-screen-training')/name,drive_root=arm)
                options.update(changes)
                command=[sys.executable,'-u','-m','homeworks.hw2.train']
                for key,value in options.items():
                    flag='--'+key.replace('_','-')
                    command.extend([flag] if value is True else [flag,str(value)])
                command.extend(['--restart-schedule','--predict-at-checkpoint'])
                (arm/'launch.json').write_text(json.dumps(dict(command=command),indent=2))
                with (arm/'training.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=PLAN['training_timeout_seconds'])
                checkpoint=arm/f"step-{PLAN['final_step']:06d}"
                assert (checkpoint/'results.complete.json').exists()
                provenance=json.loads((checkpoint/'predictions.provenance.json').read_text())
                assert provenance['num_beams']==1 and provenance['syntax_retry_beams']==0
                results[name]={}
                for stage,offset,rows in [('pilot',256,256),('confirmation',512,512)]:
                    output=arm/(stage+'.json')
                    command=[sys.executable,'-m','homeworks.hw2.numeric_eval','--model-dir',str(checkpoint),'--cache-dir',str(CACHE),'--output',str(output),'--offset',str(offset),'--rows',str(rows),'--beams','1']
                    deferred=PLAN.get('overlap_cpu_scoring',False)
                    with output.with_suffix('.log').open('w') as log:
                        subprocess.run(command+(['--generate-only'] if deferred else []),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                    if deferred:
                        scoring.start(name,stage,command+['--score-only'],output)
                    else:
                        results[name][stage]=json.loads(output.read_text())
                scoring.collect(results)
                (RUN/'progress.json').write_text(json.dumps({'completed_arms':[a for a,s in results.items() if len(s)==2],'results':results},indent=2))
            scoring.collect(results,wait=True)
            (RUN/'progress.json').write_text(json.dumps({'completed_arms':list(results),'results':results},indent=2))
            selection=(selection_fn or select_candidates)(results)
            (RUN/'selection.json').write_text(json.dumps(selection,indent=2))
            status['completed']=True
        finally:
            scoring.close()
            (RUN/'scoring-execution.json').write_text(json.dumps(scoring.completed,indent=2))
            monitor.terminate();monitor.wait(timeout=10)
            status['elapsed_seconds']=time.monotonic()-began
            (RUN/'execution.json').write_text(json.dumps(status,indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.log'}:z.write(path,path.relative_to(RUN))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true')
    run() if parser.parse_args().gpu else prepare()
