"""Export an immutable mature-growth checkpoint from Drive for bot scoring."""
import argparse
import json
from pathlib import Path

from .checkpoint_io import sha256
from .collect_scale_checkpoint import collect


def collect_mature(folder, destination, evidence, ready_path, parent, attachment, arm):
    ready = json.loads(ready_path.read_text())
    assert ready['plan']['name'] == 'mature-growth-300s-v1' and ready['gpu_launch_ready']
    identity = ready['parent_verification']['arm_checkpoints'][arm]
    assert sha256(parent/'checkpoint.complete.json') == identity['checkpoint_manifest_sha256']
    config = json.loads((parent/'run_config.json').read_text())
    # This is the existing full-stream schedule, not the pilot's wall-time limit.
    coverage_plan = dict(final_step=config['steps'], options=dict(
        batch_size=config['batch_size'], coverage_rows_per_shard=config['coverage_rows_per_shard']))
    result = collect(folder, destination, evidence, ready_path, parent, attachment,
                     coverage_plan=coverage_plan)
    actual = json.loads((destination/'run_config.json').read_text())
    assert actual['model_config']['d_ff'] == (1536 if arm == 'control42' else 3072)
    assert actual['resume_from'] == '/content/mature-growth-inputs/'+arm
    assert not actual['reset_optimizer'] and not actual['restart_schedule']
    assert actual['schedule_start'] == config['schedule_start'] == 92456
    budget = json.loads((destination/'training-budget.json').read_text())
    assert budget['max_training_seconds'] == 300 and result['step'] > 122962
    result.update(arm=arm, parent_model_sha256=identity['model_sha256'],
                  source_commit=ready['source_commit'], matched_training_seconds=budget['training_seconds'])
    (evidence/'collection.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--arm', choices=['control42', 'grown66'], required=True)
    for name in ['destination', 'evidence', 'ready-path', 'parent', 'attachment']:
        p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(collect_mature(**vars(p.parse_args())), indent=2))
