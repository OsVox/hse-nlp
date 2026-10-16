"""Resume the confirmed recipe to one complete pass through eligible rows."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

from . import learning_curve
from .learning_curve_confirm import paired_selection
from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, copy_verified, restore_data
from .tokenizer_pilot import training_command

NAME = 'selected-full-pass-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
INPUTS = Path('/content/selected-full-pass-inputs')


def selected_recipe(prior, confirmation=None):
    verified = learning_curve.select(prior['results'])
    if verified['provisional_winner'] != prior['provisional_winner']:
        raise ValueError('Inconsistent first-seed selection')
    if verified['second_seed_required']:
        if confirmation is None:
            raise ValueError('The provisional alternative needs its second-seed result')
        checked = paired_selection(confirmation['results'], prior)
        if checked['full_pass_recipe'] != confirmation['full_pass_recipe']:
            raise ValueError('Inconsistent second-seed selection')
        return checked['full_pass_recipe']
    return 'adamw42'


def build_plan(prior, confirmation, parent_config, parent_budget, eligible_rows):
    recipe = selected_recipe(prior, confirmation)
    options = dict(learning_curve.OPTIONS, **learning_curve.CONFIGS[recipe])
    # Always continue seed42: choosing a lucky seed would reuse validation scores.
    for key, value in options.items():
        if key not in {'coverage_index'} and parent_config.get(key) != value:
            raise ValueError(f'Parent recipe differs at {key}')
    step = parent_budget['optimizer_step']
    if not (0 < step < options['steps'] and parent_budget['time_limit_reached']):
        raise ValueError('Parent must be the completed first-seed pilot')
    seconds_per_update = parent_budget['training_seconds']/parent_budget['completed_updates']
    remaining = options['steps']-step
    checkpoint_steps = list(range((step//10000+1)*10000, options['steps'], 10000))+[options['steps']]
    # Include measured-speed headroom and checkpoint/export time in the hard bound.
    training_timeout = math.ceil(remaining*seconds_per_update*1.35 + len(checkpoint_steps)*180 + 300)
    options.update(max_training_seconds=None, checkpoint_training_seconds=None,
                   save_every=10000, eval_every=options['steps'], coverage_index=str(INPUTS/'index'))
    return dict(recipe=recipe, options=options, parent_step=step, parent_seed=42,
        final_step=options['steps'], eligible_rows=eligible_rows, checkpoint_steps=checkpoint_steps,
        measured_seconds_per_update=seconds_per_update,
        estimated_remaining_training_seconds=remaining*seconds_per_update,
        training_timeout_seconds=training_timeout, outer_timeout_seconds=training_timeout+900,
        public_test_used_for_selection=False,
        purpose='Complete exactly one traversal of all eligible original rows, with complete questions and programs.',
        resume_policy='Preserve optimizer, schedule, random states and consumed-row cursor from seed42. No restart or new sweep.',
        export_policy='Save and verify each retained checkpoint on Drive, export strict greedy outputs, then evaluate in the bot.',
        limitations='Validation groups remain excluded. A complete pass is not proof of optimal size or score. T4 timing remains to be verified.')


def verify_full_pass(record, plan):
    r = record['report']
    assert record['optimizer_step'] == plan['final_step']
    assert r['start_step'] == 0 and record['state']['cursor'] == [1, 0, 0]
    assert r['completed_full_passes'] == 1
    assert r['eligible_rows'] == r['presentations'] == r['unique_rows_since_tracking_started'] == plan['eligible_rows']
    assert r['repeated_presentations_since_tracking_started'] == 0
    assert r['unique_visited_by_shard'] == r['presentations_by_shard'] == r['eligible_by_shard']
    assert r['source_length'] is None and r['target_length'] is None
    assert r['untruncated_tokens'] == record['state']['untruncated_tokens']
    return r


def launcher(plan):
    return (learning_curve.LAUNCH.replace(learning_curve.NAME, NAME)
        .replace('homeworks.hw2.learning_curve', 'homeworks.hw2.selected_full_pass')
        .replace('timeout=3900', f"timeout={plan['outer_timeout_seconds']}")
        .replace('MATCHED_CURVE_STARTED', 'SELECTED_FULL_PASS_STARTED')
        .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
                 f"Hard bound {plan['outer_timeout_seconds']} seconds; resume the selected recipe through one full pass."))


def run():
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    drive_preflight(ROOT)
    ready = json.loads((PREPARED/'ready.json').read_text())
    plan = ready['plan']
    assert ready['cpu_verified'] and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip() == ready['source_commit']
    RUN.mkdir(); copy_verified(PREPARED/'ready.json', RUN/'runtime-manifest.json')
    started = time.monotonic(); status = dict(completed=False, gpu=torch.cuda.get_device_name())
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','--loop=1'], stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            INPUTS.mkdir(exist_ok=False)
            for label, source, digest in [
                ('fresh', Path(ready['fresh_holdout']['drive_path']), ready['fresh_holdout']['sha256']),
                ('parent', PREPARED/'parent.tar', ready['parent_archive_sha256'])]:
                archive = INPUTS/(label+'.tar'); copy_verified(source, archive)
                assert sha256(archive) == digest
                with tarfile.open(archive) as z: z.extractall(INPUTS, filter='data')
            parent = INPUTS/'parent'
            assert sha256(parent/'model.safetensors') == ready['parent_model_sha256']
            assert sha256(INPUTS/'index/manifest.json') == ready['fresh_preparation']['index_manifest_sha256']
            options = dict(plan['options'], resume_from=parent, cache_dir=CACHE,
                output_dir=Path('/content/selected-full-pass-training'), drive_root=RUN)
            command = training_command(options)
            (RUN/'launch.json').write_text(json.dumps(dict(command=command), indent=2))
            with (RUN/'training.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=plan['training_timeout_seconds'])
            checkpoint = RUN/f"step-{plan['final_step']:06d}"
            coverage = verify_full_pass(json.loads((checkpoint/'coverage.json').read_text()), plan)
            assert (checkpoint/'results.complete.json').exists()
            budget = json.loads((RUN/'compute-budget.json').read_text())
            assert budget['checkpoint_steps'] == plan['checkpoint_steps']
            (RUN/'full-pass-verification.json').write_text(json.dumps(coverage, indent=2))
            status['completed'] = True
        finally:
            monitor.terminate(); monitor.wait(timeout=10)
            status['elapsed_seconds'] = time.monotonic()-started
            (RUN/'execution.json').write_text(json.dumps(status, indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(RUN.rglob('*')):
                    if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.log'}:
                        z.write(p,p.relative_to(RUN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args(); run()
