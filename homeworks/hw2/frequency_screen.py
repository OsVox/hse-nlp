"""Bounded inverse-frequency trials from the verified scratch-trained best model."""
import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import tarfile
import time

from . import broad_screen
from .checkpoint_io import drive_preflight, sha256
from .question_frequency import FrequencyIndex
from .short_pilots import ROOT, CACHE, copy_verified, restore_data

NAME='frequency-screen-1k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PLAN=json.loads((Path(__file__).parent/'experiments'/NAME/'plan.json').read_text())
INDEX=Path('/content/frequency-screen-index')
LAUNCH=(broad_screen.LAUNCH.replace('broad-screen-1k-v1',NAME)
    .replace('homeworks.hw2.broad_screen','homeworks.hw2.frequency_screen')
    .replace('timeout=2400','timeout=1200')
    .replace('SHORT_SCREEN_STARTED','FREQUENCY_SCREEN_STARTED')
    .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
             'Hard bound 1200 seconds; three matched 1024-update arms only.'))


def restore_index():
    INDEX.mkdir(exist_ok=False)
    source=ROOT/'runs/frequency-index-cpu-v1/frequency-index.tar.gz'
    copy_verified(source, INDEX/'index.tar.gz')
    assert sha256(INDEX/'index.tar.gz')==PLAN['frequency_index_archive_sha256']
    with tarfile.open(INDEX/'index.tar.gz') as archive:
        archive.extractall(INDEX,filter='data')
    assert sha256(INDEX/'index/manifest.json')==PLAN['frequency_index_manifest_sha256']
    return FrequencyIndex(INDEX/'index',CACHE)


def select(results):
    result=broad_screen.select_candidates(results)
    for row in result['ranking']:
        row['eligible']=row['eligible'] and min(row['split_gains'])>0
    result['shortlist']=[r['arm'] for r in result['ranking'] if r['eligible']]
    result['training_updates_per_arm']=1024
    return result


def prepare():
    import torch
    from transformers import T5ForConditionalGeneration
    from .data import training_batches
    assert not Path('/dev/nvidia0').exists() and not torch.cuda.is_available()
    assert not (PREPARED/'ready.json').exists() and not RUN.exists()
    drive_preflight(ROOT)
    began=time.monotonic()
    bootstrap=json.loads((PREPARED/'bootstrap.json').read_text())
    assert bootstrap['plan']==PLAN
    restore_data(bootstrap['dataset'])
    index=restore_index()
    local=Path('/content/frequency-parent-preparation');local.mkdir(exist_ok=False)
    parent=local/'parent';parent.mkdir()
    source=Path(PLAN['parent_mount_path'])
    manifest=json.loads((source/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==PLAN['parent_model_sha256']
    for name,digest in manifest['files'].items():
        assert Path(name).name==name and sha256(source/name)==digest
        copy_verified(source/name,parent/name)
    copy_verified(source/'checkpoint.complete.json',parent/'checkpoint.complete.json')
    torch.set_num_threads(2)
    model=T5ForConditionalGeneration.from_pretrained(parent,local_files_only=True)
    assert sum(p.numel() for p in model.parameters())==42481152
    optimizer=torch.optim.AdamW(model.parameters(),lr=PLAN['learning_rate'])
    state=torch.load(parent/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['step']==PLAN['parent_step']
    optimizer.load_state_dict(state['optimizer'])
    for parameter,values in optimizer.state.items():
        assert values['exp_avg'].shape==parameter.shape==values['exp_avg_sq'].shape
        assert torch.isfinite(values['exp_avg']).all() and torch.isfinite(values['exp_avg_sq']).all()
    model.eval()
    with torch.inference_mode():
        assert torch.isfinite(model(input_ids=torch.tensor([[3,1]]),labels=torch.tensor([[4,1]])).loss)
    del model,optimizer,state
    streams={}
    for arm,options in PLAN['arms'].items():
        start=time.monotonic();counts=Counter();rows=0
        batches=training_batches(CACHE,128,seed=PLAN['seed']+PLAN['parent_step'],
            mixed_rows_per_shard=4096,frequency_index=index,frequency_power=options['frequency_power'])
        for batch in islice(batches,1024):
            assert len(batch)==128
            rows+=len(batch);counts.update(row['question'] for row in batch)
        assert rows==131072
        streams[arm]=dict(rows=rows,unique_questions=len(counts),seconds=time.monotonic()-start)
        print('STREAM',arm,streams[arm],flush=True)
    with tarfile.open(local/'parent.tar','w') as archive:archive.add(parent,arcname='parent')
    copy_verified(local/'parent.tar',PREPARED/'parent.tar')
    (PREPARED/'launch.py').write_text(LAUNCH)
    preflight=dict(parent_verified=True,optimizer_verified=True,forward_finite=True,
        parent_step=72000,streams=streams,elapsed_seconds=time.monotonic()-began,gpu=False,
        index_manifest_sha256=sha256(INDEX/'index/manifest.json'))
    (PREPARED/'cpu-preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    ready=dict(cpu_verified=True,source_commit=bootstrap['source_commit'],plan=PLAN,
        dataset=bootstrap['dataset'],parent_model_sha256=PLAN['parent_model_sha256'],
        parent_sha256=sha256(PREPARED/'parent.tar'),source_sha256=sha256(PREPARED/'source.tar'),
        launcher_sha256=sha256(PREPARED/'launch.py'),preflight=preflight)
    assert ready['source_sha256']==bootstrap['source_sha256']
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    print('FREQUENCY_CPU_READY',json.dumps(preflight),flush=True)


def run():
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['plan']==PLAN
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    restore_data(ready['dataset'])
    restore_index()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true')
    run() if p.parse_args().gpu else prepare()
