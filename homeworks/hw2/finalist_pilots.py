"""Continue two screened scratch models to 6k; preserve every public export."""
import argparse
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from .checkpoint_io import drive_preflight, sha256
from .short_pilots import CACHE, ROOT, TRIALS, copy_verified, restore_data

PREPARED = ROOT / 'prepared/finalist-pilots-v1'
RUN = ROOT / 'runs/finalist-pilots-v1'
PARENTS = ROOT / 'runs/short-pilots-v1'


def verify_parent(directory):
    manifest = json.loads((directory / 'checkpoint.complete.json').read_text())
    assert manifest['status'] == 'checkpoint_verified'
    for name, digest in manifest['files'].items():
        assert Path(name).name == name and sha256(directory / name) == digest
    assert json.loads((directory / 'step.json').read_text())['step'] == 2000
    assert (directory / 'results.complete.json').exists()
    return manifest


def prepare(names):
    import torch
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    assert len(names) == 2 and len(set(names)) == 2
    trials = {trial['name']: trial for trial in TRIALS}
    assert set(names).issubset(trials)
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists()
    original = json.loads((PARENTS / 'plan.json').read_text())
    completed = json.loads((PARENTS / 'results.json').read_text())
    assert len(completed) == 4, 'Inspect the complete four-way screen first'
    parents = {n: verify_parent(PARENTS / n / 'step-002000') for n in names}
    restore_data(original['dataset'])
    PREPARED.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    Path('source_commit.txt').write_text(commit + '\n')
    bundle = Path('/content/finalist-source.tar')
    with tarfile.open(bundle, 'w') as archive:
        archive.add('source_commit.txt')
        archive.add('homeworks/hw2/tokenizer')
        for path in sorted(Path('homeworks/hw2').glob('*.py')):
            archive.add(path)
    copy_verified(bundle, PREPARED / 'source.tar')
    ready = dict(names=names, parents=parents, dataset=original['dataset'],
                 source_commit=commit, source_sha256=sha256(bundle), cpu_verified=True,
                 steps=6000, start_step=2000, batch_size=128, seed=42,
                 warmup=200, save_every=6000, eval_every=500,
                 schedule='Restart cosine for 4k added updates; retain Adam moments',
                 data_stream='seed 2042 for both finalists; new deterministic stream',
                 selection='Held-out loss, numeric agreement, measured time; bot confirmation',
                 full_training_started=False)
    (PREPARED / 'ready.json').write_text(json.dumps(ready, indent=2))
    print('FINALISTS_CPU_READY', json.dumps(ready), flush=True)


def export_evidence():
    """Save downloadable evidence before the notebook releases the GPU."""
    with zipfile.ZipFile(RUN / 'exports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.log'}:
                archive.write(path, path.relative_to(RUN))


def run():
    import torch
    drive_preflight(ROOT)
    ready = json.loads((PREPARED / 'ready.json').read_text())
    assert ready['cpu_verified'] and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
    restore_data(ready['dataset'])
    RUN.mkdir(parents=True)
    copy_verified(PREPARED / 'ready.json', RUN / 'plan.json')
    trials = {trial['name']: trial for trial in TRIALS}
    results = []
    started = time.monotonic()
    with (RUN / 'gpu-utilization.csv').open('w') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits', '--loop=1'], stdout=telemetry)
        try:
            for name in ready['names']:
                destination = RUN / name
                destination.mkdir()
                parent = PARENTS / name / 'step-002000'
                assert verify_parent(parent) == ready['parents'][name]
                options = dict(steps=6000, batch_size=128, accumulation=1, seed=42,
                               warmup=200, eval_every=500, save_every=6000,
                               precision='bf16', group_window=32, mixed_rows_per_shard=4096,
                               input_pipeline='encoded', prefetch_batches=2,
                               learning_rate=trials[name]['learning_rate'], resume_from=parent,
                               cache_dir=CACHE, output_dir=Path('/content/finalist-pilots') / name,
                               drive_root=destination)
                command = [sys.executable, '-u', '-m', 'homeworks.hw2.train']
                for key, value in options.items():
                    command.extend(['--' + key.replace('_', '-'), str(value)])
                command.extend(['--restart-schedule', '--predict-at-checkpoint'])
                (destination / 'launch.json').write_text(json.dumps(dict(command=command, gpu=torch.cuda.get_device_name(), source_commit=ready['source_commit']), indent=2))
                t0 = time.monotonic()
                with (destination / 'training.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                elapsed = time.monotonic() - t0
                checkpoint = destination / 'step-006000'
                assert (checkpoint / 'results.complete.json').exists()
                command = [sys.executable, '-m', 'homeworks.hw2.numeric_eval', '--model-dir', str(checkpoint), '--cache-dir', str(CACHE), '--output', str(destination / 'numeric.json')]
                with (destination / 'numeric.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                results.append(dict(name=name, train_and_export_seconds=elapsed,
                    validation=json.loads((checkpoint / 'validation.json').read_text()),
                    numeric=json.loads((destination / 'numeric.json').read_text()), bot_status='pending'))
                (RUN / 'results.json').write_text(json.dumps(results, indent=2))
                print('FINALIST_COMPLETE', json.dumps(results[-1]), flush=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            (RUN / 'execution.json').write_text(json.dumps(dict(completed_trials=len(results), elapsed_seconds=time.monotonic()-started, full_training_started=False), indent=2))
            export_evidence()
    assert len(results) == 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--names', nargs=2)
    args = parser.parse_args()
    run() if args.gpu else prepare(args.names)
