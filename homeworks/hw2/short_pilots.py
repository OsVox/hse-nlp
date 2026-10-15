"""Drive-first, bounded scratch-model screening; never launches a full run."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .checkpoint_io import drive_preflight, sha256

ROOT = Path('/content/drive/MyDrive/HSE_NLP_HW2')
PREPARED = ROOT / 'prepared/short-pilots-v1'
RUN = ROOT / 'runs/short-pilots-v1'
CACHE = Path('/content/hw2_cache')
TRIALS = [
    dict(name='small-lr1e3', d_model=256, d_ff=768, encoder_layers=4, decoder_layers=4, num_heads=4, learning_rate=.001),
    dict(name='wide-lr1e3', d_model=512, d_ff=1536, encoder_layers=4, decoder_layers=4, num_heads=8, learning_rate=.001),
    dict(name='encoder6-lr1e3', d_model=512, d_ff=1536, encoder_layers=6, decoder_layers=4, num_heads=8, learning_rate=.001),
    dict(name='encoder6-lr3e4', d_model=512, d_ff=1536, encoder_layers=6, decoder_layers=4, num_heads=8, learning_rate=.0003),
]
PLAN = dict(trials=TRIALS, steps=2000, batch_size=128, accumulation=1,
            seed=42, warmup=200, eval_every=500, save_every=2000,
            precision='bf16', group_window=32, mixed_rows_per_shard=4096,
            input_pipeline='encoded', prefetch_batches=2,
            selection='Compare held-out numeric agreement, token loss and elapsed GPU seconds; bot scores are confirmation, not training targets.',
            next_stage='Extend the best two to 6000 only after inspecting all four results; no automatic full run.',
            soft_budget_seconds=1800, per_trial_timeout_seconds=900,
            numeric_rows=[256, 512], final_audit_rows=[512, 1024])


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256(source) == sha256(target), str(target)


def restore_data(record):
    CACHE.mkdir(exist_ok=True)
    files = record['parquet_files']
    if not all((CACHE / n).exists() and sha256(CACHE / n) == h for n, h in files.items()):
        bundle = Path('/content/pilot-data.tar')
        copy_verified(ROOT / 'prepared/data.tar', bundle)
        assert sha256(bundle) == record['data_sha256']
        with tarfile.open(bundle) as archive:
            archive.extractall(CACHE, filter='data')
    assert all(sha256(CACHE / n) == h for n, h in files.items())


def prepare():
    import torch
    from transformers import PreTrainedTokenizerFast
    from .train import new_model
    from .data import validation_examples
    assert not Path('/dev/nvidia0').exists() and not torch.cuda.is_available()
    drive_preflight(ROOT)
    assert not PREPARED.exists() and not RUN.exists(), 'Inspect previous preparation/run first'
    dataset = json.loads((ROOT / 'prepared/ready.json').read_text())
    restore_data(dataset)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer', local_files_only=True)
    counts = {}
    torch.set_num_threads(2)
    for trial in TRIALS:
        model = new_model(tokenizer, **{k:v for k,v in trial.items() if k not in ['name', 'learning_rate']})
        with torch.inference_mode():
            value = model(input_ids=torch.tensor([[3, 1]]), labels=torch.tensor([[4, 1]])).loss
            assert torch.isfinite(value)
        counts[trial['name']] = sum(p.numel() for p in model.parameters())
        del model
    assert len(validation_examples(CACHE)) == 1024
    PREPARED.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    Path('source_commit.txt').write_text(commit+'\n')
    bundle = Path('/content/short-pilots-source.tar')
    with tarfile.open(bundle, 'w') as archive:
        archive.add('source_commit.txt')
        archive.add('homeworks/hw2/tokenizer')
        for path in sorted(Path('homeworks/hw2').glob('*.py')):
            archive.add(path)
    copy_verified(bundle, PREPARED / 'source.tar')
    ready = dict(plan=PLAN, parameter_counts=counts, source_commit=commit,
                 source_sha256=sha256(bundle), dataset=dataset, cpu_verified=True)
    (PREPARED / 'ready.json').write_text(json.dumps(ready, indent=2))
    assert json.loads((PREPARED / 'ready.json').read_text()) == ready
    print('SHORT_PILOTS_CPU_READY', json.dumps(ready), flush=True)


def run():
    import torch
    drive_preflight(ROOT)
    ready = json.loads((PREPARED / 'ready.json').read_text())
    assert ready['cpu_verified'] and ready['plan'] == PLAN
    assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3, 'Use a >=40 GiB training GPU; T4 timing is separate'
    assert not RUN.exists(), 'Inspect previous results before restarting'
    restore_data(ready['dataset'])
    RUN.mkdir(parents=True)
    copy_verified(PREPARED / 'ready.json', RUN / 'plan.json')
    started = time.monotonic()
    results = []
    with (RUN / 'gpu-utilization.csv').open('w') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits', '--loop=1'], stdout=telemetry)
        try:
            for trial in TRIALS:
                if time.monotonic() - started >= PLAN['soft_budget_seconds']:
                    break
                destination = RUN / trial['name']
                destination.mkdir()
                command = [sys.executable, '-u', '-m', 'homeworks.hw2.train']
                options = {k:PLAN[k] for k in ['steps','batch_size','accumulation','seed','warmup','eval_every','save_every','precision','group_window','mixed_rows_per_shard','input_pipeline','prefetch_batches']}
                options.update({k:v for k,v in trial.items() if k != 'name'})
                options.update(cache_dir=CACHE, output_dir=Path('/content/short-pilots')/trial['name'], drive_root=destination)
                for key, value in options.items():
                    command.extend(['--'+key.replace('_','-'), str(value)])
                command.append('--predict-at-checkpoint')
                (destination / 'launch.json').write_text(json.dumps(dict(command=command, gpu=torch.cuda.get_device_name(), source_commit=ready['source_commit']), indent=2))
                t0 = time.monotonic()
                with (destination / 'training.log').open('w') as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=PLAN['per_trial_timeout_seconds'])
                assert completed.returncode == 0, f"Training failed: {trial['name']}"
                checkpoint = destination / 'step-002000'
                assert (checkpoint / 'results.complete.json').exists()
                train_seconds = time.monotonic() - t0
                command = [sys.executable, '-m', 'homeworks.hw2.numeric_eval', '--model-dir', str(checkpoint), '--cache-dir', str(CACHE), '--output', str(destination/'numeric.json')]
                with (destination / 'numeric.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                result = dict(name=trial['name'], parameters=ready['parameter_counts'][trial['name']], train_and_export_seconds=train_seconds,
                              validation=json.loads((checkpoint/'validation.json').read_text()), numeric=json.loads((destination/'numeric.json').read_text()), bot_status='pending')
                results.append(result)
                (RUN / 'results.json').write_text(json.dumps(results, indent=2))
                print('PILOT_COMPLETE', json.dumps(result), flush=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            (RUN/'execution.json').write_text(json.dumps(dict(completed_trials=len(results), elapsed_seconds=time.monotonic()-started, full_training_started=False), indent=2))
    assert len(results) == len(TRIALS), 'Screen stopped at budget; inspect completed trials'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    run() if args.gpu else prepare()
