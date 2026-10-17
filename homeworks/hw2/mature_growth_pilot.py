"""Bounded same-parent comparison of the mature model and wider feed-forward layers."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile

from .adamw_size_curve import STAGE_IDENTITY
from .async_scoring import ScoringJobs
from .checkpoint_io import drive_preflight, sha256
from .collect_scale_checkpoint import check_decoding
from .learning_curve import LAUNCH as BASE_LAUNCH, comparison
from .prepare_mature_growth import plan_from_parent
from .scaled_training import evaluate
from .short_pilots import ROOT, CACHE, copy_verified, restore_data
from .tokenizer_pilot import training_command

NAME = 'mature-growth-300s-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
INPUTS = Path('/content/mature-growth-inputs')
LAUNCH = (BASE_LAUNCH.replace('matched-curve-v1', NAME)
    .replace('homeworks.hw2.learning_curve', 'homeworks.hw2.mature_growth_pilot')
    .replace('timeout=3900', 'timeout=1800')
    .replace('MATCHED_CURVE_STARTED', 'MATURE_GROWTH_STARTED')
    .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
             'Hard bound1800seconds; two paired arms,300training seconds each, two checkpoints per arm.')
    .replace('full_process', 'growth_process').replace('full_log', 'growth_log')
    .replace('full_task', 'growth_task').replace('finish_full', 'finish_growth')
    .replace('/content/broad-screen-source', '/content/mature-growth-source')
    .replace("assert ready['cpu_verified']", "assert ready['cpu_verified'] and ready['gpu_launch_ready']")
    .replace("    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],check=True,timeout=180)\n", ''))


def options_for(plan, parent, arm, local):
    return dict(steps=184912, batch_size=128, accumulation=1, seed=42,
        warmup=0, learning_rate=3e-5, save_every=184912, eval_every=184912,
        max_training_seconds=plan['measured_training_seconds_per_arm'],
        checkpoint_training_seconds=plan['checkpoint_training_seconds'],
        precision='bf16', group_window=32, input_pipeline='encoded', prefetch_batches=2,
        coverage_index=INPUTS/'index', coverage_rows_per_shard=4096,
        untruncated_training=True, microbatch_token_budget=40960,
        microbatch_attention_budget=9961472, numeric_token_weight=1.,
        prediction_batch_size=64, prediction_max_input_tokens=0,
        prediction_max_new_tokens=1280, optimizer='adamw', restart_schedule=False,
        resume_from=parent, cache_dir=CACHE, output_dir=local, drive_root=arm)


def select(results, parent):
    if set(results) != {'control42', 'grown66'}:
        raise ValueError('Both paired final checkpoints are required')
    for values in [*results.values(), parent]:
        for stage in ('pilot', 'confirmation'):
            for key in STAGE_IDENTITY:
                if values[stage][key] != parent[stage][key]:
                    raise ValueError(f'Incomparable held-out result: {stage}/{key}')
    paired = comparison(results['grown66'], results['control42'])
    return dict(results=results, paired_comparison=paired,
        changes_from_parent={arm: comparison(values, parent) for arm, values in results.items()},
        provisional_winner='grown66' if paired['passes'] else 'control42',
        confirmation_required=paired['passes'], long_run_started=False,
        public_test_used_for_selection=False)


def checked_checkpoint(checkpoint, step, parent, d_ff):
    marker = json.loads((checkpoint/'checkpoint.complete.json').read_text())
    for name, digest in marker['files'].items():
        assert Path(name).name == name and sha256(checkpoint/name) == digest
    coverage = json.loads((checkpoint/'coverage.json').read_text())
    old = json.loads((parent/'coverage.json').read_text())
    timing = json.loads((checkpoint/'training-budget.json').read_text())
    config = json.loads((checkpoint/'run_config.json').read_text())
    assert coverage['optimizer_step'] == timing['optimizer_step'] == step
    assert config['model_config']['d_ff'] == d_ff and config['optimizer'] == 'adamw'
    assert config['schedule_start'] == 92456 and config['steps'] == 184912
    assert coverage['report']['unique_rows_since_tracking_started'] == old['report']['unique_rows_since_tracking_started']
    assert coverage['report']['presentations'] > old['report']['presentations']
    assert coverage['report']['source_length'] is None and coverage['report']['target_length'] is None
    check_decoding(json.loads((checkpoint/'predictions.provenance.json').read_text()))
    assert (checkpoint/'results.complete.json').exists()
    return dict(step=step, coverage=coverage['report'], timing=timing)


def check_budget(budget, points, parent_step, parent_presentations):
    """Reject partial, duplicate, or mismatched training intervals."""
    assert budget['stop_reason'] == 'training_time_budget'
    assert 300 <= budget['training_seconds'] < 310
    steps = budget['checkpoint_steps']
    assert len(steps) == 2 and parent_step < steps[0] < steps[1] == budget['final_step']
    assert [p['step'] for p in points] == steps
    assert budget['updates'] == steps[-1]-parent_step
    for point, expected in zip(points, (150, 300)):
        timing = point['timing']
        assert expected <= timing['training_seconds'] < expected+10
        assert timing['completed_updates'] == point['step']-parent_step
        assert timing['examples_processed'] == point['coverage']['presentations']-parent_presentations
    assert points[-1]['timing']['training_seconds'] == budget['training_seconds']
    assert points[-1]['timing']['examples_processed'] == budget['examples_processed']


def cpu_preflight():
    """Verify mounted inputs and runtime without allocating an accelerator."""
    import torch
    from .runtime_guard import verify_runtime_versions
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists()
    ready = json.loads((PREPARED/'ready.json').read_text())
    assert not ready['gpu_launch_ready'] and not RUN.exists()
    # Colab CPU images use the CPU wheel of the same PyTorch release.
    # The GPU runner still requires the exact CUDA build recorded in ready.
    cpu_versions = dict(ready['runtime_versions'])
    cpu_versions['torch'] = cpu_versions['torch'].split('+')[0]+'+cpu'
    verify_runtime_versions(cpu_versions)
    assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
    drive_preflight(ROOT)
    restore_data(ready['dataset'])
    INPUTS.mkdir(exist_ok=False)
    for source, digest in [(PREPARED/'paired-parents.tar', ready['parent_verification']['archive_sha256']),
                          (Path(ready['fresh_holdout']['drive_path']), ready['fresh_holdout']['sha256'])]:
        archive = INPUTS/source.name
        copy_verified(source, archive)
        assert sha256(archive) == digest
        with tarfile.open(archive) as tar:
            tar.extractall(INPUTS, filter='data')
    assert sha256(INPUTS/'questions.jsonl') == ready['fresh_preparation']['questions_sha256']
    assert sha256(INPUTS/'index/manifest.json') == ready['fresh_preparation']['index_manifest_sha256']
    for arm, identity in ready['parent_verification']['arm_checkpoints'].items():
        parent = INPUTS/arm
        assert sha256(parent/'checkpoint.complete.json') == identity['checkpoint_manifest_sha256']
        marker = json.loads((parent/'checkpoint.complete.json').read_text())
        for name, digest in marker['files'].items():
            assert Path(name).name == name and sha256(parent/name) == digest
        state = torch.load(parent/'training_state.pt', map_location='cpu', weights_only=False)
        assert state['step'] == ready['plan']['parent_step']
        del state
    ready.update(cpu_verified=True, gpu_launch_ready=True,
        runtime_cpu_preflight=dict(completed=True, torch_version=torch.__version__,
                                   timestamp=time.time(), all_input_hashes_verified=True))
    temporary = PREPARED/'ready.checked.json'
    temporary.write_text(json.dumps(ready, indent=2)+'\n')
    temporary.replace(PREPARED/'ready.json')
    assert json.loads((PREPARED/'ready.json').read_text()) == ready
    print('MATURE_GROWTH_CPU_READY', ready['source_commit'], flush=True)


def run():
    import torch
    from .runtime_guard import verify_runtime_versions
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    drive_preflight(ROOT)
    ready = json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['gpu_launch_ready'] and not RUN.exists()
    verify_runtime_versions(ready['runtime_versions'])
    assert torch.cuda.get_device_name() == ready['expected_gpu']
    assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3
    assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
    plan = ready['plan']; proof = ready['parent_verification']
    RUN.mkdir(); copy_verified(PREPARED/'ready.json', RUN/'runtime-manifest.json')
    began = time.monotonic(); status = dict(completed=False, gpu=torch.cuda.get_device_name())
    scoring = ScoringJobs(); curves = {}; budgets = {}
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits', '--loop=1'], stdout=telemetry)
        try:
            restore_data(ready['dataset']); INPUTS.mkdir(exist_ok=False)
            for source, digest in [(PREPARED/'paired-parents.tar', proof['archive_sha256']),
                                  (Path(ready['fresh_holdout']['drive_path']), ready['fresh_holdout']['sha256'])]:
                archive = INPUTS/source.name; copy_verified(source, archive)
                assert sha256(archive) == digest
                with tarfile.open(archive) as tar: tar.extractall(INPUTS, filter='data')
            assert sha256(INPUTS/'questions.jsonl') == ready['fresh_preparation']['questions_sha256']
            assert sha256(INPUTS/'index/manifest.json') == ready['fresh_preparation']['index_manifest_sha256']
            config = json.loads((INPUTS/'control42/run_config.json').read_text())
            assert plan == plan_from_parent(config, 122962, proof['parent_model_sha256'])
            for arm_name in plan['arms']:
                parent = INPUTS/arm_name
                identity = proof['arm_checkpoints'][arm_name]
                assert sha256(parent/'checkpoint.complete.json') == identity['checkpoint_manifest_sha256']
                assert sha256(parent/'model.safetensors') == identity['model_sha256']
                assert sha256(parent/'training_state.pt') == identity['training_state_sha256']
                arm = RUN/arm_name; arm.mkdir()
                options = options_for(plan, parent, arm, Path('/content/mature-growth-training')/arm_name)
                command = training_command(options)
                (arm/'launch.json').write_text(json.dumps(dict(command=command), indent=2))
                with (arm/'training.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=plan['per_arm_timeout_seconds'])
                budget = json.loads((arm/'compute-budget.json').read_text()); budgets[arm_name] = budget
                assert budget['stop_reason'] == 'training_time_budget' and 300 <= budget['training_seconds'] < 310
                assert len(budget['checkpoint_steps']) == 2 and budget['final_step'] == budget['checkpoint_steps'][-1]
                curves[arm_name] = []
                for step in budget['checkpoint_steps']:
                    checkpoint = arm/f'step-{step:06d}'
                    point = checked_checkpoint(checkpoint, step, parent, 1536 if arm_name == 'control42' else 3072)
                    point['results'] = evaluate(checkpoint, INPUTS/'questions.jsonl', scoring)
                    curves[arm_name].append(point)
                    (RUN/'progress.json').write_text(json.dumps(dict(curves=curves, budgets=budgets), indent=2))
                initial = json.loads((parent/'coverage.json').read_text())['report']
                check_budget(budget, curves[arm_name], plan['parent_step'], initial['presentations'])
            final = {arm: points[-1]['results'] for arm, points in curves.items()}
            (RUN/'selection.json').write_text(json.dumps(select(final, ready['parent_heldout']), indent=2))
            status.update(completed=True, total_measured_training_seconds=sum(b['training_seconds'] for b in budgets.values()))
        finally:
            scoring.close(); monitor.terminate(); monitor.wait(timeout=10)
            (RUN/'scoring-execution.json').write_text(json.dumps(scoring.completed, indent=2))
            status['elapsed_seconds'] = time.monotonic()-began
            (RUN/'execution.json').write_text(json.dumps(status, indent=2))
            with zipfile.ZipFile(RUN/'exports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv', '.log'}:
                        archive.write(path, path.relative_to(RUN))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', action='store_true', required=True)
    p.parse_args(); run()
