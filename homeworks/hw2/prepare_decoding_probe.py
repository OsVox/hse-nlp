"""Prepare the bounded 180k beam-search test locally, without a Colab runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

MODEL_HASH = '0399a924eea6ef872aa0b403b77fc5e33d24909cc7690536880bc00b254362c7'
INPUTS_HASH = '910a18cfc0485cb90c12147a9f89d3963acc9684208c7cd3e5d77817d7b9e8d4'

LAUNCH = '''from google.colab import drive, runtime
try:
    import torch
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name()
    drive.mount('/content/drive', timeout_ms=90000)
    import os, sys, json, hashlib, tarfile, subprocess, asyncio, signal, time
    from pathlib import Path
    root=Path('/content/drive/MyDrive/HSE_NLP_HW2')
    prepared=root/'prepared/beam4-180k-v1'
    ready=json.loads((prepared/'ready.json').read_text())
    assert ready['cpu_verified']
    work=Path('/content/beam4-source'); work.mkdir(exist_ok=False)
    source=(prepared/'source.tar').read_bytes()
    assert hashlib.sha256(source).hexdigest()==ready['source_sha256']
    (work/'source.tar').write_bytes(source)
    with tarfile.open(work/'source.tar') as archive: archive.extractall(work,filter='data')
    assert (work/'source_commit.txt').read_text().strip()==ready['source_commit']
    inputs=(root/'prepared/inference-speed-180k-v1/inputs.tar').read_bytes()
    assert hashlib.sha256(inputs).hexdigest()==ready['inputs_sha256']
    (work/'inputs.tar').write_bytes(inputs)
    with tarfile.open(work/'inputs.tar') as archive: archive.extractall(work/'inputs',filter='data')
    subprocess.run([sys.executable,'-m','pip','-q','install','transformers==4.57.6','pyarrow>=18'],check=True,timeout=180)
    os.environ['USE_TF']='0'; os.environ['USE_FLAX']='0'
    decoding_log=(prepared/'execution.log').open('w')
    command=[sys.executable,'-u','-m','homeworks.hw2.decoding_t4',
        '--model-dir',str(work/'inputs/model'),'--cache-dir',str(work/'inputs/cache'),
        '--output-dir',str(root/'benchmarks/beam4-180k-v1'),
        '--expected-model-sha256',ready['model_sha256'],'--beams','4','--batch-size','128']
    decoding_process=subprocess.Popen(command,cwd=work,stdout=decoding_log,stderr=subprocess.STDOUT,start_new_session=True)
    async def finish_decoding():
        status={'pid':decoding_process.pid,'started_utc':time.time()}
        try:
            await asyncio.wait_for(asyncio.to_thread(decoding_process.wait),timeout=600)
            status['returncode']=decoding_process.returncode
        except BaseException as error:
            status['error']=type(error).__name__+': '+str(error)
            if decoding_process.poll() is None:
                os.killpg(decoding_process.pid,signal.SIGTERM)
                try: await asyncio.wait_for(asyncio.to_thread(decoding_process.wait),timeout=10)
                except asyncio.TimeoutError: os.killpg(decoding_process.pid,signal.SIGKILL)
        finally:
            decoding_log.close(); status['finished_utc']=time.time()
            try: (prepared/'execution.complete.json').write_text(json.dumps(status,indent=2))
            finally: runtime.unassign()
    decoding_task=asyncio.create_task(finish_decoding())
    print('T4_DECODING_STARTED',decoding_process.pid,'600-second bound; verify runtime release after completion.')
except BaseException:
    runtime.unassign()
    raise
'''


def prepare(inputs_tar, output):
    import torch
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    from .checkpoint_io import sha256
    assert not torch.cuda.is_available()
    assert sha256(inputs_tar) == INPUTS_HASH
    assert not output.exists()
    torch.set_num_threads(2)
    comparison = json.loads(Path('homeworks/hw2/experiments/decoding-180k/comparison.json').read_text())
    assert comparison['model_sha256'] == MODEL_HASH and comparison['additional_correct'] > 0
    with tempfile.TemporaryDirectory(prefix='hw2-beam-preflight-') as tmp:
        local = Path(tmp)
        with tarfile.open(inputs_tar) as archive:
            archive.extractall(local,filter='data')
        model_dir = local/'model'
        manifest = json.loads((model_dir/'checkpoint.complete.json').read_text())
        assert sha256(model_dir/'model.safetensors') == MODEL_HASH
        for name,digest in manifest['files'].items():
            assert Path(name).name == name
            if (model_dir/name).exists(): assert sha256(model_dir/name) == digest
        questions = pq.read_table(local/'cache/test.parquet',columns=['question'])['question'].to_pylist()
        assert len(questions) == 1919
        assert hashlib.sha256(json.dumps(questions,ensure_ascii=False).encode()).hexdigest() == '214746f8d4bc918b6fa1fde8ca3185a518a60637f239a2c45efa9a8143f612d3'
        model = T5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True).eval()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir,local_files_only=True)
        batch = tokenizer(questions[:2],padding=True,return_tensors='pt',return_token_type_ids=False)
        with torch.inference_mode(): model.generate(**batch,num_beams=4,max_new_tokens=4)
    subprocess.run(['git','diff','--exit-code','HEAD','--','homeworks/hw2/*.py'],check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    output.mkdir(parents=True)
    with tarfile.open(output/'source.tar','w') as archive:
        content = (commit+'\n').encode(); info = tarfile.TarInfo('source_commit.txt');info.size=len(content)
        archive.addfile(info,io.BytesIO(content))
        for path in sorted(Path('homeworks/hw2').glob('*.py')): archive.add(path)
    compile(LAUNCH,'launch.py','exec')
    (output/'launch.py').write_text(LAUNCH)
    ready=dict(cpu_verified=True,source_commit=commit,source_sha256=sha256(output/'source.tar'),
        launcher_sha256=sha256(output/'launch.py'),model_sha256=MODEL_HASH,inputs_sha256=INPUTS_HASH,
        questions=1919,beams=4,batch_size=128,precision='fp32',max_new_tokens=192,
        required_gpu='T4',passes=2,per_pass_limit_seconds=180,process_limit_seconds=600,
        heldout_comparison=comparison,training=False)
    (output/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(ready,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs-tar',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.inputs_tar,args.output)
