import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
from pathlib import Path
assert full_process.poll() is None, 'Main training must still be live'
assert len(os.sched_getaffinity(0))>=8
root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
prepared=root/'prepared/candidate-vote-180k-v1'
work=Path('/content/candidate-vote-source');assert work.exists()
assert (work/'source_commit.txt').read_text().strip()=='f047d7d79e58e679866d5571f45d07da0ef8be38'
assert hashlib.sha256((work/'inputs/model/model.safetensors').read_bytes()).hexdigest()=='0399a924eea6ef872aa0b403b77fc5e33d24909cc7690536880bc00b254362c7'
run=root/'benchmarks/candidate-vote-180k-v1';assert run.exists()
cpus=sorted(os.sched_getaffinity(0))[-4:]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',USE_TF='0',USE_FLAX='0')
vote_log=(run/'confirmation.log').open('w')
cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m','homeworks.hw2.candidate_vote','--model-dir',str(work/'inputs/model'),'--cache-dir','/content/hw2_cache','--output',str(run/'confirmation'),'--offset','768','--rows','256','--beams','4','--batch-size','32']
(run/'confirmation-launch.json').write_text(json.dumps({'command':cmd,'source_commit':'f047d7d79e58e679866d5571f45d07da0ef8be38','cpu_only':True,'cpu_affinity':cpus,'timeout_seconds':360},indent=2))
vote_process=subprocess.Popen(cmd,cwd=work,env=env,stdout=vote_log,stderr=subprocess.STDOUT,start_new_session=True)
async def finish_vote(vote_process=vote_process, vote_log=vote_log, run=run):
    status={'pid':vote_process.pid,'started_utc':time.time()}
    try:
        await asyncio.wait_for(asyncio.to_thread(vote_process.wait),timeout=360)
        status['returncode']=vote_process.returncode
    except BaseException as error:
        status['error']=type(error).__name__+': '+str(error)
        if vote_process.poll() is None:
            os.killpg(vote_process.pid,signal.SIGTERM)
            try:await asyncio.wait_for(asyncio.to_thread(vote_process.wait),timeout=5)
            except asyncio.TimeoutError:os.killpg(vote_process.pid,signal.SIGKILL)
    finally:
        vote_log.close();status['finished_utc']=time.time()
        (run/'confirmation.complete.json').write_text(json.dumps(status,indent=2))
vote_task=asyncio.create_task(finish_vote())
print('CPU_VOTE_CONFIRMATION_STARTED',vote_process.pid,'Main GPU training continues')
