"""CPU preparation and a bounded 42M continuation after the measured pilots."""
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

NAME = 'full-42m-100k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PARENT = ROOT/'runs/grouped-42m-20k-v1/continuation/step-020000'
MODEL_HASH = '10b1897916d111c8c79d22cca52a147de91b6360b4de617d7c000ea236d7b67a'
PLAN = dict(parent_step=20000, final_step=100000, batch_size=128,
            learning_rate=.00015, warmup=500, save_every=10000, eval_every=1000,
            precision='bf16', group_window=32, mixed_rows_per_shard=4096,
            input_pipeline='encoded', prefetch_batches=2, prediction_batch_size=256,
            added_example_presentations=80000*128, total_example_presentations=100000*128,
            data_policy='Original supplied train rows; question-level held-out exclusion. Counts are presentations, not unique rows.',
            timing_policy='User deferred T4 inference requirement until quality is improved.',
            rate_rationale='Half the prior peak to reduce restart disruption; this is a hypothesis, not a tuned optimum.',
            training_timeout_seconds=8500, outer_timeout_seconds=9000,
            review='Score every 10k checkpoint once; inspect the first two before allowing the whole remaining budget to run unattended.')

LAUNCH = '''from google.colab import drive, runtime
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3
    drive.mount('/content/drive', timeout_ms=180000)
    import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
    from pathlib import Path
    prepared=Path('/content/drive/MyDrive/HSE_NLP_HW2/prepared/full-42m-100k-v1')
    ready=json.loads((prepared/'ready.json').read_text())
    assert ready['cpu_verified']
    work=Path('/content/full-42m-source');work.mkdir(exist_ok=False)
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as z:z.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],check=True,timeout=180)
    full_log=(prepared/'execution.log').open('w')
    full_process=subprocess.Popen([sys.executable,'-u','-m','homeworks.hw2.full_training','--gpu'],cwd=work,stdout=full_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish_full(full_process=full_process, full_log=full_log, prepared=prepared):
        status={'pid':full_process.pid,'started_utc':time.time()}
        try:
            await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=9000)
            status['returncode']=full_process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if full_process.poll() is None:
                os.killpg(full_process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=10)
                except asyncio.TimeoutError:os.killpg(full_process.pid,signal.SIGKILL)
        finally:
            full_log.close();status['finished_utc']=time.time()
            try:(prepared/'execution.complete.json').write_text(json.dumps(status,indent=2))
            finally:runtime.unassign()
    full_task=asyncio.create_task(finish_full())
    print('FULL_TRAINING_STARTED',full_process.pid,'Hard bound 9000 seconds; inspect each saved checkpoint.')
except BaseException:
    runtime.unassign()
    raise
'''


def prepare():
    import torch
    from transformers import T5ForConditionalGeneration
    assert not Path('/dev/nvidia0').exists() and not torch.cuda.is_available()
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists(), 'Inspect prior preparation before retrying'
    receipt = json.loads((PARENT/'bot_receipt.json').read_text())
    assert receipt['status']=='scored' and receipt['public_correct']==320
    assert receipt['model_sha256']==MODEL_HASH
    dataset = json.loads((ROOT/'prepared/ready.json').read_text())
    restore_data(dataset)
    local = Path('/content/full-42m-preparation');local.mkdir(exist_ok=False)
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
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2))
    assert json.loads((PREPARED/'ready.json').read_text())==ready
    print('FULL_CPU_READY',json.dumps(ready),flush=True)


def run():
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory>=40*1024**3
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['plan']==PLAN and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    RUN.mkdir(parents=True);copy_verified(PREPARED/'ready.json',RUN/'plan.json')
    began=time.monotonic();status=dict(completed=False,gpu=torch.cuda.get_device_name())
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            local=Path('/content/full-42m-inputs');local.mkdir(exist_ok=False)
            copy_verified(PREPARED/'parent.tar',local/'parent.tar')
            assert sha256(local/'parent.tar')==ready['parent_sha256']
            with tarfile.open(local/'parent.tar') as z:z.extractall(local,filter='data')
            options={k:PLAN[k] for k in ['batch_size','learning_rate','warmup','save_every','eval_every','precision','group_window','mixed_rows_per_shard','input_pipeline','prefetch_batches','prediction_batch_size']}
            options.update(resume_from=local/'parent',steps=PLAN['final_step'],accumulation=1,seed=42,cache_dir=CACHE,output_dir=Path('/content/full-42m-training'),drive_root=RUN)
            command=[sys.executable,'-u','-m','homeworks.hw2.train']
            for key,value in options.items():command.extend(['--'+key.replace('_','-'),str(value)])
            command.extend(['--restart-schedule','--predict-at-checkpoint'])
            (RUN/'launch.json').write_text(json.dumps(dict(command=command),indent=2))
            with (RUN/'training.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=PLAN['training_timeout_seconds'])
            checkpoint=RUN/f"step-{PLAN['final_step']:06d}"
            assert (checkpoint/'results.complete.json').exists()
            with (RUN/'numeric.log').open('w') as log:
                subprocess.run([sys.executable,'-m','homeworks.hw2.numeric_eval','--model-dir',str(checkpoint),'--cache-dir',str(CACHE),'--output',str(RUN/'numeric.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            status['completed']=True
        finally:
            monitor.terminate();monitor.wait(timeout=10)
            status['elapsed_seconds']=time.monotonic()-began
            (RUN/'execution.json').write_text(json.dumps(status,indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.log'}:z.write(path,path.relative_to(RUN))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true')
    run() if parser.parse_args().gpu else prepare()
