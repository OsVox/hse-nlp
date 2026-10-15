"""CPU preparation and bounded continuation from our Drive step-60,000 model."""
from pathlib import Path
import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from zipfile import ZipFile, ZIP_DEFLATED

from .checkpoint_io import sha256, drive_preflight

ROOT = Path('/content/drive/MyDrive/HSE_NLP_HW2')
PREPARED = ROOT/'prepared/continuation-60k-v1'
PARENT_RUN = 'continue-20k-60k-b128-v1'
PARENT_HASH = 'f1b4feaefc78ce7455d0fa51fabd9bfc0e9f900eb0ed39c1e58768a9ea45f865'
RUN = 'continue-60k-180k-b128-v1'


def copy_verified(source, destination, expected=None):
    expected = expected or sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected:
        return
    temporary = destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(source, temporary)
    assert sha256(temporary) == expected, str(destination)
    temporary.replace(destination)


def restore_inputs():
    cache = Path('/content/hw2_cache')
    record = json.loads((ROOT/'prepared/ready.json').read_text())
    cache.mkdir(exist_ok=True)
    if not all((cache/name).exists() and sha256(cache/name) == value
               for name, value in record['parquet_files'].items()):
        archive = Path('/content/data.tar')
        copy_verified(ROOT/'prepared/data.tar', archive, record['data_sha256'])
        with tarfile.open(archive) as z:
            z.extractall(cache, filter='data')
    assert all(sha256(cache/name) == value for name, value in record['parquet_files'].items())
    parent = ROOT/'runs'/PARENT_RUN/'step-060000'
    manifest = json.loads((parent/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors'] == PARENT_HASH
    local_parent = Path('/content/parent-060000')
    for name, value in manifest['files'].items():
        assert Path(name).name == name
        copy_verified(parent/name, local_parent/name, value)
    copy_verified(parent/'checkpoint.complete.json', local_parent/'checkpoint.complete.json')
    pool_ready = json.loads((ROOT/'prepared/continuation-v1/ready.json').read_text())
    pool = Path('/content/probe.jsonl')
    copy_verified(ROOT/'prepared/continuation-v1/probe.jsonl', pool, pool_ready['pool_sha256'])
    return cache, local_parent, pool


def prepare():
    import torch
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists(), 'Prepare on CPU'
    drive_preflight(ROOT)
    assert not (ROOT/'runs'/RUN).exists(), 'Inspect existing run before repeating'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    cache, parent, pool = restore_inputs()
    PREPARED.mkdir(parents=True,exist_ok=True)
    from .benchmark_input import preflight
    preflight(parent, pool, PREPARED/'input-equivalence.json', 32768)
    # Reopen our own model on CPU to catch incompatible files before GPU allocation.
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(parent,local_files_only=True)
    assert sum(p.numel() for p in model.parameters()) == 9967360
    del model
    # Persist the final receipts from the previous completed run without browser upload.
    source_root = Path('homeworks/hw2')
    evidence_root = ROOT/'runs'/PARENT_RUN
    paths = sorted((source_root/'submissions').glob('2026-10-15-continue-b128-step-*.json'))
    assert len(paths) == 8
    for path in paths:
        copy_verified(path, evidence_root/'receipts'/path.name)
    for name in ['summary.json','utilization-summary.json']:
        copy_verified(source_root/'experiments'/PARENT_RUN/name,evidence_root/name)
    for name in ['report.pdf','hw2_inference_colab.ipynb']:
        copy_verified(source_root/name,evidence_root/name)
    bundle = Path('/content/continuation-60k-source.tar')
    with tarfile.open(bundle,'w') as z:
        payload = (commit+'\n').encode()
        info = tarfile.TarInfo('source_commit.txt'); info.size = len(payload)
        z.addfile(info,io.BytesIO(payload))
        for path in sorted(source_root.glob('*.py')):
            z.add(path,arcname=str(path))
    copy_verified(bundle,PREPARED/'source.tar')
    ready = {'source_commit':commit,'source_sha256':sha256(bundle),
             'parent_model_sha256':PARENT_HASH,'pool_sha256':sha256(pool),
             'input_equivalence_sha256':sha256(PREPARED/'input-equivalence.json'),
             'gpu_allocated':False,'parent_step':60000,'final_step':180000,
             'previous_run_receipts_copied':8}
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2))
    assert json.loads((PREPARED/'ready.json').read_text()) == ready
    print('CPU_60K_CONTINUATION_READY',json.dumps(ready),flush=True)


def run_gpu():
    """One child process: restore, probe, train, archive, then request release."""
    from google.colab import runtime
    run_root = ROOT/'runs'/RUN
    telemetry = monitor = None
    try:
        import torch
        drive_preflight(ROOT)
        ready = json.loads((PREPARED/'ready.json').read_text())
        assert ready['gpu_allocated'] is False
        assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
        assert sha256(PREPARED/'input-equivalence.json') == ready['input_equivalence_sha256']
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        cache,parent,pool = restore_inputs()
        assert sha256(pool) == ready['pool_sha256']
        run_root.mkdir(parents=True,exist_ok=False)
        benchmark_root = ROOT/'benchmarks/input-from-60k-v1'
        from .benchmark_input import benchmark
        benchmark(parent,pool,benchmark_root,PREPARED/'input-equivalence.json',32768)
        selection = json.loads((benchmark_root/'selection.json').read_text())
        chosen = selection['chosen']
        pipeline = 'legacy' if chosen == 'legacy' else 'encoded'
        prefetch = 2 if chosen == 'encoded-prefetch' else 0
        command = [sys.executable,'-u','-m','homeworks.hw2.train','--resume-from',str(parent),
                   '--restart-schedule','--steps','180000','--batch-size','128','--accumulation','1',
                   '--precision','bf16','--group-window','32','--mixed-rows-per-shard','4096',
                   '--learning-rate','0.0003','--warmup','500','--save-every','20000','--eval-every','20000',
                   '--input-pipeline',pipeline,'--prefetch-batches',str(prefetch),
                   '--cache-dir',str(cache),'--output-dir',f'/content/{RUN}',
                   '--drive-root',str(run_root),'--predict-at-checkpoint']
        launch = {'command':command,'source_commit':ready['source_commit'],'parent_model_sha256':PARENT_HASH,
                  'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'gpu':torch.cuda.get_device_name(0),'input_selection':selection}
        (run_root/'launch.json').write_text(json.dumps(launch,indent=2))
        telemetry = (run_root/'gpu-utilization.csv').open('w')
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        with (run_root/'training.log').open('w') as log:
            rc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        monitor.terminate(); monitor.wait(); telemetry.close()
        monitor = telemetry = None
        markers = sorted(run_root.glob('step-*/results.complete.json'))
        with ZipFile(run_root/'predictions-and-evidence.zip','w',ZIP_DEFLATED) as z:
            for marker in markers:
                for name in ['predictions.csv','predictions.jsonl','predictions.provenance.json',
                             'results.complete.json','validation.json','checkpoint.complete.json','run_config.json']:
                    path = marker.parent/name
                    if path.exists(): z.write(path,arcname=f'{marker.parent.name}/{name}')
            for name in ['launch.json','training.log','gpu-utilization.csv']:
                z.write(run_root/name,arcname=name)
        complete = {'returncode':rc,'checkpoints':len(markers),
                    'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (run_root/'execution.complete.json').write_text(json.dumps(complete,indent=2))
        assert rc == 0 and len(markers) == 6, complete
    except BaseException as error:
        if run_root.exists():
            (run_root/'execution.error.txt').write_text(type(error).__name__+': '+str(error))
        raise
    finally:
        if monitor is not None:
            monitor.terminate(); monitor.wait()
        if telemetry is not None:
            telemetry.close()
        try:
            runtime.unassign()
        except Exception as error:
            (PREPARED/'release-error.txt').write_text(str(error))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    args = parser.parse_args()
    run_gpu() if args.gpu else prepare()
