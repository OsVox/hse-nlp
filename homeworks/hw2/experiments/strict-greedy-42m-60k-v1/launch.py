import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
from pathlib import Path
assert full_process.poll() is None
assert len(os.sched_getaffinity(0))>=8
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
prepared=root/'prepared/hw2-strict-greedy-v1'
work=Path('/content/strict-greedy-60k-source');work.mkdir(exist_ok=False)
b=(prepared/'source.tar').read_bytes()
assert hashlib.sha256(b).hexdigest()=='d71f1028136da126f03e78ff1cda90de16e7a19f846cb394fde803d1193a3a29'
(work/'source.tar').write_bytes(b)
with tarfile.open(work/'source.tar') as t:t.extractall(work,filter='data')
assert (work/'source_commit.txt').read_text().strip()=='8dcb74d2164a30c7b3169dcf7adf76114139761f'
model=root/'runs/full-42m-100k-v1/step-060000'
assert json.loads((model/'results.complete.json').read_text())['step']==60000
assert hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest()==json.loads((model/'checkpoint.complete.json').read_text())['files']['model.safetensors']
run=root/'benchmarks/strict-greedy-42m-60k-v1';run.mkdir(exist_ok=False)
cpus=sorted(os.sched_getaffinity(0))[-4:]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',USE_TF='0',USE_FLAX='0')
greedy_log=(run/'execution.log').open('w')
cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m','homeworks.hw2.greedy_export','--model-dir',str(model),'--cache-dir','/content/hw2_cache','--output-dir',str(run/'step-060000')]
(run/'launch.json').write_text(json.dumps({'command':cmd,'source_commit':'8dcb74d2164a30c7b3169dcf7adf76114139761f','cpu_only':True,'cpu_affinity':cpus,'timeout_seconds':600},indent=2))
greedy_process=subprocess.Popen(cmd,cwd=work,env=env,stdout=greedy_log,stderr=subprocess.STDOUT,start_new_session=True)
async def finish_greedy(greedy_process=greedy_process,greedy_log=greedy_log,run=run):
    status={'pid':greedy_process.pid,'started_utc':time.time()}
    try:
        await asyncio.wait_for(asyncio.to_thread(greedy_process.wait),timeout=600)
        status['returncode']=greedy_process.returncode
    except BaseException as error:
        status['error']=type(error).__name__+': '+str(error)
        if greedy_process.poll() is None:
            os.killpg(greedy_process.pid,signal.SIGTERM)
            try:await asyncio.wait_for(asyncio.to_thread(greedy_process.wait),timeout=5)
            except asyncio.TimeoutError:os.killpg(greedy_process.pid,signal.SIGKILL)
    finally:
        greedy_log.close();status['finished_utc']=time.time()
        (run/'execution.complete.json').write_text(json.dumps(status,indent=2))
greedy_task=asyncio.create_task(finish_greedy())
print('CPU_STRICT_GREEDY_STARTED',greedy_process.pid,'GPU training continues')
