from google.colab import drive,runtime
try:
    import torch
    assert 'A100' in torch.cuda.get_device_name(), 'Wrong GPU: stop before setup'
    drive.mount('/content/drive',timeout_ms=180000)
    import os,sys,json,hashlib,subprocess,tarfile,asyncio,signal,time
    from pathlib import Path
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and hasattr(torch.optim,'Muon')
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
    prepared=root/'prepared/muon-batch-a100-v1'
    ready=json.loads((prepared/'ready.json').read_text())
    work=Path('/content/muon-batch-probe');work.mkdir(exist_ok=False)
    payload=(root/'prepared/muon-batch-probe-v1/inputs.tar').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==ready['inputs_sha256']
    (work/'inputs.tar').write_bytes(payload)
    with tarfile.open(work/'inputs.tar') as z:z.extractall(work,filter='data')
    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18'],check=True,timeout=180)
    output=root/'benchmarks/muon-batch-a100-v1'
    assert not output.exists()
    local_log=work/'execution.log';full_log=local_log.open('w')
    command=[sys.executable,'-u','-m','homeworks.hw2.benchmark_muon_batches','--pool','pool.jsonl','--tokenizer','homeworks/hw2/tokenizer','--gate','gate.json','--output',str(output)]
    process=subprocess.Popen(command,cwd=work,stdout=full_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish():
        status={'pid':process.pid,'started_utc':time.time(),'ready':ready}
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=240)
            status['returncode']=process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:await asyncio.wait_for(asyncio.to_thread(process.wait),timeout=10)
                except asyncio.TimeoutError:os.killpg(process.pid,signal.SIGKILL)
        finally:
            full_log.close();status['finished_utc']=time.time()
            try:
                (prepared/'execution.log').write_bytes(local_log.read_bytes())
                (prepared/'execution.complete.json').write_text(json.dumps(status,indent=2))
            finally:runtime.unassign()
    task=asyncio.create_task(finish())
    print('MUON_BATCH_PROBE_STARTED',process.pid,'240-second bound; discarded updates; automatic release.')
except BaseException:
    runtime.unassign()
    raise
