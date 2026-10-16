from google.colab import drive,runtime
from pathlib import Path
import hashlib,json,tarfile,subprocess,sys,asyncio,time,os,signal
assert not Path('/dev/nvidia0').exists(), 'CPU only: do not allocate GPU for this audit'
drive.mount('/content/drive',timeout_ms=300000)
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
audit_root=root/'runs/train-diversity-audit-v1'
assert not audit_root.exists(), 'Inspect existing audit before restarting'
audit_root.mkdir()
audit_source=(root/'prepared/data_diversity_audit.py').read_bytes()
assert hashlib.sha256(audit_source).hexdigest()=='a408e6ec48d0ac763530f3333aa4c46b877d96922fe6dc730069db9675fcfbc7'
(audit_root/'source.py').write_bytes(audit_source)
audit_driver=r'''
from pathlib import Path
import json,hashlib,tarfile,subprocess,sys,shutil
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
out=root/'runs/train-diversity-audit-v1'
ready=json.loads((root/'prepared/ready.json').read_text())
cache=Path('/content/hw2_cache');cache.mkdir(exist_ok=True)
bundle=Path('/content/audit-data.tar');shutil.copyfile(root/'prepared/data.tar',bundle)
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==ready['data_sha256']
with tarfile.open(bundle) as z:z.extractall(cache,filter='data')
for n,h in ready['parquet_files'].items():
 assert hashlib.sha256((cache/n).read_bytes()).hexdigest()==h
with tarfile.open(root/'prepared/broad-screen-1k-v1/parent.tar') as z:
 payload=z.extractfile('parent/tokenizer.json').read()
assert hashlib.sha256(payload).hexdigest()=='d9fa7b5618323be055ad10ebe760cff2801f251b39dabb7dc62188567c5a6414'
Path('/content/audit-tokenizer.json').write_bytes(payload)
subprocess.run([sys.executable,'-m','pip','-q','install','pyarrow','tokenizers'],check=True)
subprocess.run([sys.executable,'-u',str(out/'source.py'),'--cache',str(cache),'--tokenizer','/content/audit-tokenizer.json','--output',str(out)],check=True)
'''
Path('/content/audit_driver.py').write_text(audit_driver)
audit_log=(audit_root/'execution.log').open('w')
audit_process=subprocess.Popen([sys.executable,'-u','/content/audit_driver.py'],stdout=audit_log,stderr=subprocess.STDOUT,start_new_session=True)
async def finish_audit():
 status={'pid':audit_process.pid,'started_utc':time.time(),'gpu':False,'source_sha256':hashlib.sha256(audit_source).hexdigest()}
 try:
  await asyncio.wait_for(asyncio.to_thread(audit_process.wait),timeout=900)
  status['returncode']=audit_process.returncode
 except BaseException as error:
  status['error']=str(error)
  if audit_process.poll() is None:
   os.killpg(audit_process.pid,signal.SIGTERM)
 finally:
  audit_log.close();status['finished_utc']=time.time()
  try:(audit_root/'execution.complete.json').write_text(json.dumps(status,indent=2))
  finally:runtime.unassign()
audit_task=asyncio.create_task(finish_audit())
print('CPU_AUDIT_STARTED',audit_process.pid,'hard bound 900 seconds')
