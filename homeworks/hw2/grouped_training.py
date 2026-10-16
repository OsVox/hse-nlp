"""One prepared GPU session: speed probe, optional quality gate, 20k-equivalent stage."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, copy_verified, restore_data

PREPARED = ROOT/'prepared/grouped-42m-20k-v1'
RUN = ROOT/'runs/grouped-42m-20k-v1'
PARENT = ROOT/'runs/finalist-pilots-v1/encoder6-lr3e4/step-006000'
MODEL_HASH = '7d0898f6595ee684df6f8ffa6231bbf914bc81aee91e95298003b1ff69b66deb'
PILOT_EXAMPLES = 262144
TOTAL_EXAMPLES = 20000*128


def end_step(parent_step, batch_size, examples):
    assert examples > 0 and examples % batch_size == 0
    return parent_step + examples//batch_size


def select_quality(control, candidate):
    assert candidate['numeric']['usable_references'] == control['numeric']['usable_references']
    return (candidate['numeric']['numeric_correct'] >= control['numeric']['numeric_correct']
            and candidate['validation']['token_loss'] <= control['validation']['token_loss']*1.01)


def prepare():
    import torch
    from transformers import T5ForConditionalGeneration
    assert not Path('/dev/nvidia0').exists() and not torch.cuda.is_available()
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists()
    t4 = json.loads((ROOT/'benchmarks/t4-comparison-v1/encoder6-6k/results.json').read_text())
    assert t4['completed'] and t4['model_sha256']==MODEL_HASH
    fast = next(x for x in t4['summary'] if x['batch_size']==256)
    assert fast['max_seconds'] < 180 and fast['all_outputs_equal_baseline']
    gate = json.loads((ROOT/'prepared/production-batch-finalist6k-v1/ready.json').read_text())
    assert gate['cpu_verified'] and gate['model_sha256']==MODEL_HASH
    receipt = json.loads((PARENT/'bot_receipt.json').read_text())
    assert receipt['status']=='scored' and receipt['public_correct']==182
    assert 'T4' in t4['gpu'] and all(r.get('jsonl_sha256')==receipt['jsonl_sha256'] for r in t4['runs'])
    dataset = json.loads((ROOT/'prepared/ready.json').read_text())
    restore_data(dataset)
    local = Path('/content/grouped-42m-preparation');local.mkdir(exist_ok=False)
    parent = local/'parent';parent.mkdir()
    manifest = json.loads((PARENT/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==MODEL_HASH
    for name, digest in manifest['files'].items():
        assert Path(name).name==name and sha256(PARENT/name)==digest
        copy_verified(PARENT/name,parent/name)
    copy_verified(PARENT/'checkpoint.complete.json',parent/'checkpoint.complete.json')
    torch.set_num_threads(2)
    model=T5ForConditionalGeneration.from_pretrained(parent,local_files_only=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    state=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['step']==6000
    optimizer.load_state_dict(state['optimizer'])
    for parameter, values in optimizer.state.items():
        assert values['exp_avg'].shape==parameter.shape==values['exp_avg_sq'].shape
    del state,optimizer,model
    subprocess.run([sys.executable,'-m','unittest','homeworks.hw2.tests.test_continuation',
                    'homeworks.hw2.tests.test_grouped_training'],check=True)
    PREPARED.mkdir(parents=True)
    bundle=local/'parent.tar'
    with tarfile.open(bundle,'w') as z:z.add(parent,arcname='parent')
    copy_verified(bundle,PREPARED/'parent.tar')
    pool=ROOT/'prepared/production-batch-finalist6k-v1/probe.jsonl'
    assert sha256(pool)==gate['pool_sha256']
    copy_verified(pool,PREPARED/'probe.jsonl')
    (PREPARED/'batch-gate.json').write_text(json.dumps(gate,indent=2))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    Path('source_commit.txt').write_text(commit+'\n')
    source=local/'source.tar'
    with tarfile.open(source,'w') as z:
        z.add('source_commit.txt')
        for path in sorted(Path('homeworks/hw2').glob('*.py')):z.add(path)
    copy_verified(source,PREPARED/'source.tar')
    ready=dict(cpu_verified=True,source_commit=commit,source_sha256=sha256(source),
        parent_sha256=sha256(bundle),parent_model_sha256=MODEL_HASH,dataset=dataset,
        pool_sha256=gate['pool_sha256'],pilot_examples=PILOT_EXAMPLES,
        final_total_examples=TOTAL_EXAMPLES,initial_examples=6000*128,
        learning_rate=.0003,prediction_batch_size=256,outer_budget_seconds=2700,
        protocol='Speed probe; if batch changes, equal-count quality pilots; then 20k-equivalent stage. No full-corpus run.',
        quality_gate='Candidate numeric correct >= control and token loss <= 1.01 * control.')
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2))
    print('GROUPED_CPU_READY',json.dumps(ready),flush=True)


def train(name,parent,batch,examples,*,pilot):
    destination=RUN/name;destination.mkdir()
    start=json.loads((parent/'step.json').read_text())['step']
    end=end_step(start,batch,examples)
    options=dict(resume_from=parent,steps=end,batch_size=batch,accumulation=1,
        seed=42,learning_rate=.0003,warmup=(128 if pilot else 500)*128//batch,
        save_every=end if pilot else 5000*128//batch,
        eval_every=end if pilot else 1000*128//batch,precision='bf16',
        group_window=32,mixed_rows_per_shard=4096,input_pipeline='encoded',
        prefetch_batches=2,prediction_batch_size=256,cache_dir=CACHE,
        output_dir=Path('/content/grouped-training')/name,drive_root=destination)
    command=[sys.executable,'-u','-m','homeworks.hw2.train']
    for key,value in options.items():command.extend(['--'+key.replace('_','-'),str(value)])
    command.extend(['--restart-schedule','--predict-at-checkpoint'])
    (destination/'launch.json').write_text(json.dumps(dict(command=command,
        added_examples=examples,parent_step=start,end_step=end,batch_size=batch),indent=2))
    began=time.monotonic()
    with (destination/'training.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600 if pilot else 1800)
    checkpoint=destination/f'step-{end:06d}'
    assert (checkpoint/'results.complete.json').exists()
    with (destination/'numeric.log').open('w') as log:
        subprocess.run([sys.executable,'-m','homeworks.hw2.numeric_eval','--model-dir',str(checkpoint),
            '--cache-dir',str(CACHE),'--output',str(destination/'numeric.json')],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    record=dict(name=name,checkpoint=str(checkpoint),batch_size=batch,added_examples=examples,
        elapsed_seconds=time.monotonic()-began,
        validation=json.loads((checkpoint/'validation.json').read_text()),
        numeric=json.loads((destination/'numeric.json').read_text()),bot_status='pending')
    (destination/'result.json').write_text(json.dumps(record,indent=2))
    print('TRAIN_STAGE_COMPLETE',json.dumps(record),flush=True)
    return record


def run():
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory>=40*1024**3
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    RUN.mkdir(parents=True)
    copy_verified(PREPARED/'ready.json',RUN/'plan.json')
    began=time.monotonic();records=[]
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            local=Path('/content/grouped-inputs');local.mkdir(exist_ok=False)
            copy_verified(PREPARED/'parent.tar',local/'parent.tar')
            assert sha256(local/'parent.tar')==ready['parent_sha256']
            with tarfile.open(local/'parent.tar') as z:z.extractall(local,filter='data')
            parent=local/'parent'
            copy_verified(PREPARED/'probe.jsonl',local/'probe.jsonl')
            with (RUN/'batch-probe.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-m','homeworks.hw2.benchmark_production_batches',
                    '--checkpoint',str(parent),'--pool',str(local/'probe.jsonl'),
                    '--gate',str(PREPARED/'batch-gate.json'),'--output',str(RUN/'batch-probe')],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            speed=json.loads((RUN/'batch-probe/results.json').read_text())
            assert speed['completed']
            batch=speed['selected_batch'];consumed=6000*128
            if batch!=128:
                control=train('quality-b128',parent,128,PILOT_EXAMPLES,pilot=True)
                candidate=train(f'quality-b{batch}',parent,batch,PILOT_EXAMPLES,pilot=True)
                records.extend([control,candidate])
                chosen=candidate if select_quality(control,candidate) else control
                batch=chosen['batch_size'];parent=Path(chosen['checkpoint']);consumed+=PILOT_EXAMPLES
            selection=dict(batch_size=batch,parent=str(parent),consumed_examples=consumed,
                remaining_examples=TOTAL_EXAMPLES-consumed,quality_results=records)
            (RUN/'selection.json').write_text(json.dumps(selection,indent=2))
            print('TRAINING_SELECTION',json.dumps(selection),flush=True)
            records.append(train('continuation',parent,batch,TOTAL_EXAMPLES-consumed,pilot=False))
            (RUN/'results.json').write_text(json.dumps(records,indent=2))
        finally:
            monitor.terminate();monitor.wait(timeout=10)
            (RUN/'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-began,
                completed_stages=len(records),gpu=torch.cuda.get_device_name()),indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.log'}:
                        z.write(path,path.relative_to(RUN))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true')
    run() if parser.parse_args().gpu else prepare()
