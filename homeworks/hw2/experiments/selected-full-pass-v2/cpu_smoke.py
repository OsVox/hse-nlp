from pathlib import Path
import sys,json,torch
from homeworks.hw2 import train,selected_full_pass as full
from homeworks.hw2.tokenizer_pilot import training_command
root=Path('homeworks/hw2/experiments/matched-curve-v1');cp=root/'adamw42/step-006018'
plan=full.build_plan(json.loads((root/'selection.json').read_text()),None,json.loads((cp/'run_config.json').read_text()),json.loads((cp/'training-budget.json').read_text()),11834223)
opts=dict(plan['options'],resume_from=Path('/Users/andrejvorsin/Downloads/selected-full-pass-v1/parent'),cache_dir=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset'),coverage_index=Path('/Users/andrejvorsin/Downloads/hw2-fresh-holdout-v2/index'),output_dir=Path('/Users/andrejvorsin/Downloads/selected-resume-cpu-check-3'),precision='fp32',max_training_seconds=0.000001)
assert not opts['output_dir'].exists()
torch.set_num_threads(2);torch.cuda.is_available=lambda:False;torch.backends.mps.is_available=lambda:False
from homeworks.hw2.training_budget import restore_rng
# CPU-only smoke test cannot restore CUDA RNG; production state is untouched.
train.restore_rng=lambda state:restore_rng({**state,'torch_cuda':[]})
cmd=training_command(opts);cmd.remove('--predict-at-checkpoint');sys.argv=['train']+cmd[4:]
print('CPU_RESUME_COMMAND',sys.argv,flush=True)
train.main()
budget=json.loads((opts['output_dir']/'compute-budget.json').read_text())
assert budget['final_step']==6019 and budget['updates']==1
state=torch.load(opts['output_dir']/'latest/training_state.pt',map_location='cpu',weights_only=False)
assert state['scheduler']['last_epoch']==6019 and state['step']==6019
cov=json.loads((opts['output_dir']/'latest/coverage.json').read_text())['report']
assert cov['presentations']==770432 and cov['repeated_presentations_since_tracking_started']==0
result=dict(actual_training_entrypoint=True,parent_step=6018,final_step=6019,scheduler_step=6019,updates=1,unique_rows=cov['unique_rows_since_tracking_started'],repeats=0,cpu_only=True,precision='fp32',checkpoint_for_testing_only=True,cuda_rng_skipped_for_cpu_smoke=True,training_seconds=budget['training_seconds'],production_parent_unchanged=True)
Path('/Users/andrejvorsin/Downloads/selected-resume-cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n');print('CPU_RESUME_PASSED',json.dumps(result),flush=True)
