"""Exercise two real trainer resumes of a temporary grown model on CPU.

All probe weights are temporary and discarded. The active GPU run and its
parent checkpoint remain untouched. No public predictions are generated.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
import sys
import tempfile

import torch
from transformers import T5ForConditionalGeneration

from homeworks.hw2 import train
from homeworks.hw2.checkpoint_io import archive_checkpoint, sha256
from homeworks.hw2.grow_ffn import grow_feed_forward, grow_adamw_optimizer
from homeworks.hw2.tokenizer_pilot import training_command
from homeworks.hw2.training_budget import restore_rng


signal.alarm(300)
torch.set_num_threads(2)
assert not torch.cuda.is_available()
torch.backends.mps.is_available = lambda: False
train.restore_rng = lambda state: restore_rng({**state, 'torch_cuda': []})
parent = Path('/Users/andrejvorsin/Downloads/hw2-selected-scale-v1/step-122962')
manifest = json.loads((parent/'checkpoint.complete.json').read_text())
for name, digest in manifest['files'].items():
    assert sha256(parent/name) == digest
coverage = json.loads((parent/'coverage.json').read_text())
config = json.loads((parent/'run_config.json').read_text())
state = torch.load(parent/'training_state.pt', map_location='cpu', weights_only=False)
model = T5ForConditionalGeneration.from_pretrained(parent, local_files_only=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
optimizer.load_state_dict(state['optimizer'])
grown = grow_feed_forward(model, 3072)
grown_optimizer = grow_adamw_optimizer(model, grown, optimizer)
grown_state = dict(state, optimizer=grown_optimizer.state_dict())
records = []
with tempfile.TemporaryDirectory(prefix='hw2-growth-resume-') as temporary:
    root = Path(temporary)
    raw = root/'grown-raw'; raw.mkdir()
    grown.save_pretrained(raw, safe_serialization=True)
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'coverage.json', 'step.json']:
        shutil.copyfile(parent/name, raw/name)
    new_config = deepcopy(config)
    new_config['model_config'] = grown.config.to_dict()
    new_config['growth_preparation'] = dict(parent_model_sha256=manifest['files']['model.safetensors'],
        parent_training_state_sha256=manifest['files']['training_state.pt'],
        original_d_ff=1536, grown_d_ff=3072, cpu_probe_only=True)
    (raw/'run_config.json').write_text(json.dumps(new_config, indent=2)+'\n')
    torch.save(grown_state, raw/'training_state.pt')
    resume = root/'grown-parent'
    archive_checkpoint(raw, resume)
    # Free redundant model/moment copies before the real training entrypoint.
    del model, optimizer, grown, grown_optimizer, grown_state, state
    options = json.loads(Path('homeworks/hw2/experiments/selected-scale-v1/plan.json').read_text())['options']
    options.update(max_training_seconds=.000001, precision='fp32',
        cache_dir=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset'),
        coverage_index=Path('/Users/andrejvorsin/Downloads/hw2-fresh-holdout-v2-restored/index'))
    for index in [1, 2]:
        output = root/f'interval-{index}'
        opts = dict(options, resume_from=resume, output_dir=output, restart_schedule=False)
        cmd = training_command(opts)
        if '--predict-at-checkpoint' in cmd:
            cmd.remove('--predict-at-checkpoint')
        sys.argv = ['train'] + cmd[4:]
        train.main()
        latest = output/'latest'
        updated = torch.load(latest/'training_state.pt', map_location='cpu', weights_only=False)
        budget = json.loads((output/'compute-budget.json').read_text())
        report = json.loads((latest/'coverage.json').read_text())['report']
        step = 122962 + index
        assert budget['updates'] == 1 and budget['final_step'] == step
        assert updated['step'] == step
        assert updated['scheduler']['last_epoch'] == step - config['schedule_start']
        assert all(s['step'].item() == step for s in updated['optimizer']['state'].values())
        assert report['presentations'] == coverage['report']['presentations'] + 128*index
        assert report['unique_rows_since_tracking_started'] == coverage['report']['unique_rows_since_tracking_started']
        saved_config = json.loads((latest/'config.json').read_text())
        assert saved_config['d_ff'] == 3072
        records.append(dict(step=step, scheduler_phase_step=updated['scheduler']['last_epoch'],
            optimizer_moment_step=step, presentations=report['presentations'],
            learning_rate=updated['optimizer']['param_groups'][0]['lr'],
            training_seconds=budget['training_seconds']))
        resume = root/f'archived-{index}'
        archive_checkpoint(latest, resume)
for name, digest in manifest['files'].items():
    assert sha256(parent/name) == digest
result = dict(status='passed', actual_training_entrypoint=True, parent_step=122962,
    parent_model_sha256=manifest['files']['model.safetensors'],
    grown_parameters=66074112, records=records, optimizer_history_preserved=True,
    schedule_phase_preserved=True, data_cursor_preserved=True,
    production_parent_unchanged=True, cpu_only=True, precision='fp32',
    cuda_rng_skipped_for_cpu_only=True, discarded_probe_weights=True,
    gpu_training_launched=False, quality_improvement_claimed=False,
    limitation='CPU engineering probe; not CUDA/BF16 resume equivalence or a quality pilot')
Path('homeworks/hw2/experiments/ffn-growth-cpu-v1/resume-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
