from google.colab import drive,runtime
try:
 import torch
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 assert torch.cuda.get_device_properties(0).total_memory >= 80*1024**3
 drive.mount('/content/drive',timeout_ms=180000)
 import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
 from pathlib import Path
 prepared=Path('/content/drive/MyDrive/HSE_NLP_HW2/prepared/complete-budget-gpu-v1')
 ready=json.loads((prepared/'ready.json').read_text());assert ready['cpu_verified']
 run=Path('/content/drive/MyDrive/HSE_NLP_HW2/benchmarks/complete-budget-gpu-v1')
 assert not run.exists()
 work=Path('/content/complete-budget-source');work.mkdir(exist_ok=False)
 for name in ['source.tar','pool.json','gate.json']:
  payload=(prepared/name).read_bytes();assert hashlib.sha256(payload).hexdigest()==ready['files'][name]
  (work/name).write_bytes(payload)
 with tarfile.open(work/'source.tar') as z:z.extractall(work,filter='data')
 assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
 os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
 subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18'],check=True,timeout=180)
 sys.path.insert(0,str(work))
 from homeworks.hw2.runtime_guard import finalize_run
 local_log=work/'execution.log';full_log=local_log.open('w')
 full_process=subprocess.Popen([sys.executable,'-u','-m','homeworks.hw2.benchmark_complete_budgets','--prepared',str(work),'--output',str(run)],cwd=work,stdout=full_log,stderr=subprocess.STDOUT,start_new_session=True)
 async def finish_probe():
  status={'pid':full_process.pid,'started_utc':time.time()}
  try:
   await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=360)
   status['returncode']=full_process.returncode
  except BaseException as error:
   status['error']=type(error).__name__+': '+str(error)
   if full_process.poll() is None:
    os.killpg(full_process.pid,signal.SIGTERM)
    try:await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=10)
    except asyncio.TimeoutError:os.killpg(full_process.pid,signal.SIGKILL)
  finally:finalize_run(full_log,status,prepared,local_log,runtime.unassign)
 probe_task=asyncio.create_task(finish_probe())
 print('COMPLETE_BUDGET_PROBE_STARTED',full_process.pid,'hard bound 360 seconds; discard every probe weight')
except BaseException:
 runtime.unassign()
 raise
