from google.colab import drive, runtime
from pathlib import Path
import asyncio, hashlib, json, os, signal, subprocess, sys, tarfile, time

try:
    assert not Path('/dev/nvidia0').exists(), 'CPU only'
    drive.mount('/content/drive', timeout_ms=300000)
    root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
    prepared=root/'prepared/frequency-screen-1k-v1'
    run=prepared
    assert not (prepared/'ready.json').exists(), 'Inspect existing preparation before retry'
    ready=json.loads((prepared/'bootstrap.json').read_text())
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    work=Path('/content/frequency-preparation-source');work.mkdir(exist_ok=False)
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as archive:archive.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    subprocess.run([sys.executable,'-m','pip','-q','install','pyarrow>=18','transformers==4.57.6'],check=True,timeout=180)
    driver=r'''
from homeworks.hw2.frequency_screen import prepare
prepare()
'''
    (work/'driver.py').write_text(driver)
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    audit_log=(run/'cpu-execution.log').open('w')
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
            try:(run/'cpu-execution.complete.json').write_text(json.dumps(status,indent=2))
            finally:runtime.unassign()
    audit_task=asyncio.create_task(finish_audit())
    print('FREQUENCY_PREFLIGHT_CPU_STARTED',audit_process.pid,'hard bound 1200 seconds')
except BaseException:
    runtime.unassign()
    raise
