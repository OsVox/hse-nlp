from google.colab import drive, runtime
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert 'RTX PRO 6000' in torch.cuda.get_device_name()
    assert torch.cuda.get_device_properties(0).total_memory >= 40*1024**3
    drive.mount('/content/drive', timeout_ms=180000)
    import os,sys,json,hashlib,tarfile,subprocess,asyncio,signal,time
    from pathlib import Path
    prepared=Path('/content/drive/MyDrive/HSE_NLP_HW2/prepared/target-length-1k-v1')
    ready=json.loads((prepared/'ready.json').read_text())
    assert ready['cpu_verified']
    work=Path('/content/broad-screen-source');work.mkdir(exist_ok=False)
    payload=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['source_sha256']
    (work/'source.tar').write_bytes(payload)
    with tarfile.open(work/'source.tar') as z:z.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18','requests>=2.31'],check=True,timeout=180)
    sys.path.insert(0,str(work))
    from homeworks.hw2.runtime_guard import finalize_run
    local_log=work/'execution.log'
    full_log=local_log.open('w')
    full_process=subprocess.Popen([sys.executable,'-u','-m','homeworks.hw2.target_length_pilot','--gpu'],cwd=work,stdout=full_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish_full(full_process=full_process, full_log=full_log, prepared=prepared):
        status={'pid':full_process.pid,'started_utc':time.time()}
        try:
            await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=900)
            status['returncode']=full_process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if full_process.poll() is None:
                os.killpg(full_process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(asyncio.to_thread(full_process.wait),timeout=10)
                except asyncio.TimeoutError:os.killpg(full_process.pid,signal.SIGKILL)
        finally:
            finalize_run(full_log,status,prepared,local_log,runtime.unassign)
    full_task=asyncio.create_task(finish_full())
    print('TARGET_LENGTH_STARTED',full_process.pid,'Hard bound 900 seconds; two matched 1024-update arms only.')
except BaseException:
    runtime.unassign()
    raise
