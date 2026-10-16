from pathlib import Path
import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
assert not Path('/dev/nvidia0').exists(), 'CPU preparation only'
from google.colab import drive,runtime
drive.mount('/content/drive',timeout_ms=180000)
p=Path('/content/drive/MyDrive/HSE_NLP_HW2/prepared/numeric-loss-bootstrap-v1')
r=json.loads((p/'ready.json').read_text())
assert r['source_commit']=='8755a8f4867683d940e9b34fa48365e3cf4f8c21'
w=Path('/content/numeric-loss-bootstrap');w.mkdir(exist_ok=False)
b=(p/'source.tar').read_bytes();assert hashlib.sha256(b).hexdigest()==r['source_sha256']
(w/'source.tar').write_bytes(b)
with tarfile.open(w/'source.tar') as z:z.extractall(w,filter='data')
env=os.environ.copy();env.update(USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],env=env,check=True,timeout=180)
log=(p/'execution.log').open('w')
process=subprocess.Popen([sys.executable,'-u','-m','homeworks.hw2.loss_pilot'],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
async def finish(process=process,log=log,p=p):
    status={'pid':process.pid,'cpu_only':True,'started_utc':time.time()}
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=600)
        status['returncode']=process.returncode
    except BaseException as e:
        status['error']=type(e).__name__+': '+str(e)
        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    finally:
        log.close();status['finished_utc']=time.time()
        (p/'execution.complete.json').write_text(json.dumps(status,indent=2))
        runtime.unassign()
task=asyncio.create_task(finish())
print('NUMERIC_LOSS_CPU_PREPARATION_STARTED',process.pid)
