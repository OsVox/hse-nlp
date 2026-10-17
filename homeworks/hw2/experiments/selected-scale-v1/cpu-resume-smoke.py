from pathlib import Path
from copy import deepcopy
import sys,json,torch
from homeworks.hw2 import train,learning_curve
from homeworks.hw2.tokenizer_pilot import training_command
from homeworks.hw2.checkpoint_io import sha256,archive_checkpoint
from homeworks.hw2.training_budget import restore_rng

parent=Path('/Users/andrejvorsin/Downloads/hw2-full-pass-average-cpu-v1/parents/step-092456')
manifest=json.loads((parent/'checkpoint.complete.json').read_text())
for name,digest in manifest['files'].items():assert sha256(parent/name)==digest
options=deepcopy(learning_curve.OPTIONS)
for key in ('initialization_vocab_size','encoder_layers','decoder_layers'):options.pop(key)
options.update(optimizer='adamw',steps=184912,learning_rate=.00003,warmup=0,
 save_every=184912,eval_every=184912,checkpoint_training_seconds=None,
 max_training_seconds=.000001,precision='fp32',
 cache_dir=Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset'),
 coverage_index=Path('/Users/andrejvorsin/Downloads/hw2-fresh-holdout-v2-restored/index'))
torch.set_num_threads(2);torch.cuda.is_available=lambda:False;torch.backends.mps.is_available=lambda:False
train.restore_rng=lambda state:restore_rng({**state,'torch_cuda':[]})
records=[];resume=parent
for index in (1,2):
 output=Path('/Users/andrejvorsin/Downloads/selected-scale-cpu-smoke')/f'interval-{index}'
 opts=dict(options,resume_from=resume,output_dir=output,restart_schedule=index==1)
 cmd=training_command(opts);cmd.remove('--predict-at-checkpoint');sys.argv=['train']+cmd[4:]
 if not output.exists():train.main()
 budget=json.loads((output/'compute-budget.json').read_text())
 assert budget['final_step']==92456+index and budget['updates']==1
 state=torch.load(output/'latest/training_state.pt',map_location='cpu',weights_only=False)
 assert state['scheduler']['last_epoch']==index
 assert state['step']==92456+index
 moment_steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
 assert moment_steps and min(moment_steps)==max(moment_steps)==92456+index
 lr=state['optimizer']['param_groups'][0]['lr'];assert .0000299 < lr <= .00003
 coverage=json.loads((output/'latest/coverage.json').read_text())['report']
 assert coverage['presentations']==11834223+128*index
 assert coverage['completed_full_passes']==1
 assert coverage['repeated_presentations_since_tracking_started']==128*index
 records.append(dict(step=state['step'],scheduler_phase_step=index,optimizer_moment_step=moment_steps[0],
  learning_rate=lr,presentations=coverage['presentations'],repeated_rows=coverage['repeated_presentations_since_tracking_started'],
  training_seconds=budget['training_seconds']))
 resume=output.parent/f'archived-{index}'
 archive_checkpoint(output/'latest',resume)
for name,digest in manifest['files'].items():assert sha256(parent/name)==digest
result=dict(cpu_verified=True,actual_training_entrypoint=True,cpu_only=True,
 production_parent_unchanged=True,precision='fp32',cuda_rng_skipped_for_cpu_only=True,
 original_checkpoint_step=92456,explicit_low_lr_phase=True,optimizer_moments_preserved=True,
 second_interval_restored_same_phase=True,records=records,quality_improvement_claimed=False,
 note='Tests only the possible AdamW42 continuation branch. No model selection or GPU allocation follows from this smoke test.')
Path('homeworks/hw2/experiments/selected-scale-v1/cpu-resume-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print('CPU_SCALE_RESUME_VERIFIED',json.dumps(result),flush=True)
