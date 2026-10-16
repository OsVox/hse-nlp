from google.colab import drive, runtime
from pathlib import Path
import asyncio, hashlib, json, os, signal, subprocess, sys, tarfile, time

try:
    assert not Path('/dev/nvidia0').exists(), 'CPU only'
    drive.mount('/content/drive', timeout_ms=300000)
    root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
    prepared=root/'prepared/cpu-data-tokenizer-audit-v1'
    run=root/'runs/cpu-data-tokenizer-audit-v1'
    assert not run.exists(), 'Inspect existing audit before any retry'
    ready=json.loads((prepared/'ready.json').read_text())
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    work=Path('/content/cpu-data-tokenizer-source');work.mkdir(exist_ok=False)
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as archive:archive.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    run.mkdir();(run/'ready.json').write_text(json.dumps(ready,indent=2))
    subprocess.run([sys.executable,'-m','pip','-q','install','pyarrow>=18','tokenizers'],check=True,timeout=180)
    driver=r'''
from pathlib import Path
import hashlib,json,shutil,tarfile,time,zipfile
from homeworks.hw2.python_tokenizer_audit import audit as tokenizer_audit
from homeworks.hw2.duplicate_consistency_audit import audit as consistency_audit
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
run=root/'runs/cpu-data-tokenizer-audit-v1'
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
 tokenizer_audit(cache,Path('homeworks/hw2/tokenizer'),run/'tokenizers')
 consistency_audit(cache,run/'consistency')
 status['completed']=True
finally:
 status['elapsed_seconds']=time.monotonic()-started
 (run/'execution.json').write_text(json.dumps(status,indent=2))
 with zipfile.ZipFile(run/'exports.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in sorted(run.rglob('*')):
   if path.is_file() and path.suffix in {'.json','.jsonl','.log'}:z.write(path,path.relative_to(run))
'''
    (work/'driver.py').write_text(driver)
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
    print('CPU_TOKENIZER_AUDIT_STARTED',audit_process.pid,'hard bound 1200 seconds')
except BaseException:
    runtime.unassign()
    raise
