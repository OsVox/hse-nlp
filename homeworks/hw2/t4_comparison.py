"""CPU-stage the established model and 42M finalist for bounded T4 timing."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, copy_verified

PREPARED = ROOT / 'prepared/t4-comparison-v1'
RUN = ROOT / 'benchmarks/t4-comparison-v1'
LARGE_PREPARED = ROOT / 'prepared/inference-finalist6k-v1'
LARGE_PARENT = ROOT / 'runs/finalist-pilots-v1/encoder6-lr3e4/step-006000'


def prepare():
    import torch
    from . import inference_probe_colab as probe
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists()
    baseline = ROOT / 'prepared/inference-speed-180k-v1'
    original = json.loads((baseline / 'ready.json').read_text())
    assert original['cpu_preflight_passed']
    assert sha256(baseline / 'inputs.tar') == original['inputs_sha256']
    assert original['model_sha256'] == '0399a924eea6ef872aa0b403b77fc5e33d24909cc7690536880bc00b254362c7'
    manifest = json.loads((LARGE_PARENT / 'checkpoint.complete.json').read_text())
    assert (LARGE_PARENT / 'results.complete.json').exists()
    probe.PARENT = LARGE_PARENT
    probe.MODEL_HASH = manifest['files']['model.safetensors']
    probe.PREPARED = LARGE_PREPARED
    torch.set_num_threads(2)
    probe.prepare(expected_parameters=42481152)
    large = json.loads((LARGE_PREPARED / 'ready.json').read_text())
    PREPARED.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    Path('source_commit.txt').write_text(commit + '\n')
    bundle = Path('/content/t4-comparison-source.tar')
    with tarfile.open(bundle, 'w') as archive:
        archive.add('source_commit.txt')
        for path in sorted(Path('homeworks/hw2').glob('*.py')):
            archive.add(path)
    copy_verified(bundle, PREPARED / 'source.tar')
    plan = dict(cpu_verified=True, source_commit=commit, source_sha256=sha256(bundle),
        models=[dict(name='baseline180k', prepared=str(baseline), ready=original),
                dict(name='encoder6-6k', prepared=str(LARGE_PREPARED), ready=large)],
        batches=[64, 128, 256], rounds=2, precision='fp32',
        soft_seconds_per_model=600, hard_seconds_per_model=720,
        required_gpu='T4', training=False,
        selection='Prefer a reproducibly faster batch; compare exact programs and score changed chosen outputs separately.')
    (PREPARED / 'ready.json').write_text(json.dumps(plan, indent=2))
    print('T4_COMPARISON_CPU_READY', json.dumps(plan), flush=True)


def run():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'T4' in torch.cuda.get_device_name()
    drive_preflight(ROOT)
    plan = json.loads((PREPARED / 'ready.json').read_text())
    assert plan['cpu_verified'] and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip() == plan['source_commit']
    RUN.mkdir(parents=True)
    copy_verified(PREPARED / 'ready.json', RUN / 'plan.json')
    results = []
    started = time.monotonic()
    try:
        for item in plan['models']:
            work = Path('/content/t4-models') / item['name']
            work.mkdir(parents=True, exist_ok=False)
            bundle = work / 'inputs.tar'
            copy_verified(Path(item['prepared']) / 'inputs.tar', bundle)
            assert sha256(bundle) == item['ready']['inputs_sha256']
            with tarfile.open(bundle) as archive:
                archive.extractall(work, filter='data')
            assert sha256(work / 'model/model.safetensors') == item['ready']['model_sha256']
            assert sha256(work / 'cache/test.parquet') == item['ready']['test_sha256']
            destination = RUN / item['name']
            command = [sys.executable, '-u', '-m', 'homeworks.hw2.benchmark_inference',
                '--model-dir', 'model', '--cache-dir', 'cache', '--output-dir', str(destination),
                '--expected-model-sha256', item['ready']['model_sha256'],
                '--batches', '64', '128', '256', '--budget-seconds', '600']
            env = dict(os.environ, USE_TF='0', USE_FLAX='0')
            with (RUN / (item['name'] + '.log')).open('w') as log:
                subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=720)
            record = json.loads((destination / 'results.json').read_text())
            results.append(dict(name=item['name'], completed=record.get('completed', False),
                                summary=record.get('summary', [])))
            (RUN / 'summary.json').write_text(json.dumps(results, indent=2))
            print('T4_MODEL_COMPLETE', json.dumps(results[-1]), flush=True)
    finally:
        (RUN / 'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
            completed_models=len(results), gpu=torch.cuda.get_device_name(), training=False), indent=2))
        with zipfile.ZipFile(RUN / 'exports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(RUN.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.log'}:
                    archive.write(path, path.relative_to(RUN))
    assert len(results) == 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    run() if args.gpu else prepare()
