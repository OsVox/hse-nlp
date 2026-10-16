from pathlib import Path
import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
assert not Path('/dev/nvidia0').exists(), 'Use a CPU runtime'
from google.colab import drive,runtime
drive.mount('/content/drive',timeout_ms=180000)
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
prepared=root/'prepared/checkpoint-average-70k-v1'
ready=json.loads((prepared/'ready.json').read_text())
assert ready['source_commit']=='861ae9795086a67af530f8eb2fc873423f466fbd'
work=Path('/content/checkpoint-average-source');work.mkdir(exist_ok=False)
payload=(prepared/'source.tar').read_bytes()
assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
(work/'source.tar').write_bytes(payload)
with tarfile.open(work/'source.tar') as t:t.extractall(work,filter='data')
assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
assert (root/'runs/full-42m-100k-v1/intentional-stop.json').exists()
env=os.environ.copy();env.update(USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],env=env,check=True,timeout=180)
output=root/'benchmarks/checkpoint-average-70k-v1'
assert not output.exists()
log=(prepared/'execution.log').open('w')
command=[sys.executable,'-u','-m','homeworks.hw2.average_pilot','--root',str(root),'--output',str(output)]
process=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
async def finish_average(process=process,log=log,prepared=prepared):
    status={'pid':process.pid,'started_utc':time.time(),'cpu_only':True}
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=1500)
        status['returncode']=process.returncode
    except BaseException as error:
        status['error']=type(error).__name__+': '+str(error)
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=5)
            except asyncio.TimeoutError:os.killpg(process.pid,signal.SIGKILL)
    finally:
        log.close();status['finished_utc']=time.time()
        (prepared/'execution.complete.json').write_text(json.dumps(status,indent=2))
        runtime.unassign()
average_task=asyncio.create_task(finish_average())
print('CPU_AVERAGING_STARTED',process.pid)
