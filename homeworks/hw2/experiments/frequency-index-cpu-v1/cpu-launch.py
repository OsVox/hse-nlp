from google.colab import drive, runtime
from pathlib import Path
import asyncio, hashlib, json, os, signal, subprocess, sys, tarfile, time

try:
    assert not Path('/dev/nvidia0').exists(), 'CPU only'
    drive.mount('/content/drive', timeout_ms=300000)
    root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
    prepared=root/'prepared/frequency-index-cpu-v1'
    run=root/'runs/frequency-index-cpu-v1'
    assert not run.exists(), 'Inspect existing audit before any retry'
    ready=json.loads((prepared/'ready.json').read_text())
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    work=Path('/content/frequency-index-source');work.mkdir(exist_ok=False)
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as archive:archive.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    run.mkdir();(run/'ready.json').write_text(json.dumps(ready,indent=2))
    subprocess.run([sys.executable,'-m','pip','-q','install','pyarrow>=18','transformers==4.57.6'],check=True,timeout=180)
    driver=r'''
from pathlib import Path
import hashlib,json,shutil,tarfile,time,zipfile
from homeworks.hw2.question_frequency import build, FrequencyIndex
from homeworks.hw2.data import training_batches
from itertools import islice
from collections import Counter
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
run=root/'runs/frequency-index-cpu-v1'
ready=json.loads((run/'ready.json').read_text());dataset=ready['dataset']
started=time.monotonic();status={'completed':False,'gpu':False}
try:
 cache=Path('/content/hw2_cache');cache.mkdir(exist_ok=False)
 bundle=Path('/content/cpu-audit-data.tar');shutil.copyfile(root/'prepared/data.tar',bundle)
 assert hashlib.sha256(bundle.read_bytes()).hexdigest()==dataset['data_sha256']
 with tarfile.open(bundle) as archive:archive.extractall(cache,filter='data')
 for name,digest in dataset['parquet_files'].items():
  assert Path(name).name==name and hashlib.sha256((cache/name).read_bytes()).hexdigest()==digest
 print('DATA_VERIFIED_CPU',flush=True)
 result=build(cache,run/'index')
 index=FrequencyIndex(run/'index',cache)
 checks={}
 for power in [0,.5,1]:
  start=time.monotonic();counts=Counter();h=hashlib.sha256();rows=0
  batches=training_batches(cache,128,seed=42,mixed_rows_per_shard=4096,frequency_index=index,frequency_power=power)
  for batch in islice(batches,1024):
   assert len(batch)==128
   for row in batch:
    h.update(json.dumps(row,sort_keys=True).encode());counts[row['question']]+=1;rows+=1
  checks[str(power)]={'rows':rows,'unique_questions':len(counts),'seconds':time.monotonic()-start,'stream_sha256':h.hexdigest()}
  print('STREAM',power,checks[str(power)],flush=True)
 (run/'stream-preflight.json').write_text(json.dumps(checks,indent=2))
 with tarfile.open(run/'frequency-index.tar.gz','w:gz') as archive:
  archive.add(run/'index',arcname='index')
 status['completed']=True
finally:
 status['elapsed_seconds']=time.monotonic()-started
 (run/'execution.json').write_text(json.dumps(status,indent=2))
 with zipfile.ZipFile(run/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in sorted(run.rglob('*')):
   if path.is_file() and path.suffix in {'.json','.jsonl','.log'}:z.write(path,path.relative_to(run))
'''
    (work/'driver.py').write_text(driver)
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    audit_log=(run/'execution.log').open('w')
    audit_process=subprocess.Popen([sys.executable,'-u','driver.py'],cwd=work,
        stdout=audit_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish_audit(process=audit_process, log=audit_log, run=run):
        status={'pid':process.pid,'started_utc':time.time(),'gpu':False}
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=1200)
            status['returncode']=process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=10)
                except asyncio.TimeoutError:os.killpg(process.pid,signal.SIGKILL)
        finally:
            log.close();status['finished_utc']=time.time()
            try:(run/'execution.complete.json').write_text(json.dumps(status,indent=2))
            finally:runtime.unassign()
    audit_task=asyncio.create_task(finish_audit())
    print('FREQUENCY_INDEX_CPU_STARTED',audit_process.pid,'hard bound 1200 seconds')
except BaseException:
    runtime.unassign()
    raise
