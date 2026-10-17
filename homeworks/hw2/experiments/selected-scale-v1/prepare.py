"""Materialize the selected scale-up only after the size screen is complete."""
from pathlib import Path
import json
from homeworks.hw2.scaled_training import build_plan,launcher
from homeworks.hw2.checkpoint_io import sha256

exp=Path('homeworks/hw2/experiments/selected-scale-v1')
size=Path('homeworks/hw2/experiments/adamw-size-curve-v1')
prior=json.loads((size/'selection.json').read_text())
execution=json.loads((size/'execution.json').read_text())
assert execution['completed']
assert prior['provisional_winner']=='adamw42' and not prior['second_seed_required'], 'Prepare the paired confirmation first if78M wins'
parent=Path('/Users/andrejvorsin/Downloads/hw2-full-pass-average-cpu-v1/parents/step-092456')
config=json.loads((parent/'run_config.json').read_text())
base_run=Path('homeworks/hw2/experiments/selected-full-pass-v2')
budget=json.loads((base_run/'compute-budget.json').read_text())
plan=build_plan(prior,None,config,92456,budget['training_seconds']/budget['updates'])
source=json.loads((exp/'prepared-source-local.json').read_text())
smoke=json.loads((exp/'cpu-resume-smoke.json').read_text())
assert smoke['cpu_verified'] and smoke['optimizer_moments_preserved'] and smoke['second_interval_restored_same_phase']
local=Path('/Users/andrejvorsin/Downloads/selected-scale-v1')
assert sha256(local/'source.tar')==source['sha256']
manifest=json.loads((parent/'checkpoint.complete.json').read_text())
for name,digest in manifest['files'].items():assert sha256(parent/name)==digest
ready=json.loads((size/'ready.json').read_text())
ready.update(cpu_verified=True,selection_verified=True,source_commit=source['source_commit'],
 source_sha256=source['sha256'],plan=plan,cpu_preflight=smoke,
 parent_archive_sha256=sha256(local/'parent.tar'),parent_model_sha256=manifest['files']['model.safetensors'],
 parent_heldout=json.loads((base_run/'heldout-results.json').read_text()),
 size_selection=prior,size_confirmation=None,size_selection_sha256=sha256(size/'selection.json'),
 tests=dict(scale_driver=6,continuation_and_budget=6,actual_cpu_resume_updates=2),
 scale_up_authorized=True,public_answers_used_for_training=False)
# Drop first-screen-only metadata; keep the verified dataset, holdout and versions.
for key in ('reference_sha256','cpu_averaging_selection_sha256','prerequisite_cpu_averaging_completed'):
 ready.pop(key,None)
code=launcher(plan);(exp/'launch.py').write_text(code)
ready['launcher_sha256']=sha256(exp/'launch.py')
(exp/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
(exp/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:plan[k] for k in ('recipe','parent_step','final_step','estimated_training_seconds','total_training_budget_seconds','outer_timeout_seconds','patience')},indent=2))
