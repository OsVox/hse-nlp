"""Prepare an inference experiment on CPU; run a bounded probe on one T4."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from .checkpoint_io import drive_preflight, sha256

ROOT = Path('/content/drive/MyDrive/HSE_NLP_HW2')
PREPARED = ROOT/'prepared/inference-speed-160k-v1'
PARENT = ROOT/'runs/continue-60k-180k-b128-v1/step-160000'
MODEL_HASH = '5b9c70b5d9b46271821aaf72aacd144ef64f8cc29ecf73865ff058511155d92b'
RESULTS = ROOT/'benchmarks/inference-speed-160k-t4-v1'


def prepare():
    import torch
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    assert not torch.cuda.is_available() and not Path('/dev/nvidia0').exists(), 'Prepare on CPU'
    drive_preflight(ROOT)
    assert not PREPARED.exists(), 'Inspect existing preparation before repeating'
    manifest = json.loads((PARENT/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors'] == MODEL_HASH
    selected = {n:h for n,h in manifest['files'].items() if n not in ['training_state.pt']}
    assert all(sha256(PARENT/n)==h for n,h in selected.items())
    local = Path('/content/inference-speed-preparation')
    local.mkdir(exist_ok=False)
    (local/'model').mkdir()
    for name in selected:
        shutil.copyfile(PARENT/name,local/'model'/name)
    (local/'model/checkpoint.complete.json').write_text(json.dumps(manifest,indent=2))
    dataset = json.loads((ROOT/'prepared/ready.json').read_text())
    # Extract just the already archived official test file, never fetch during GPU use.
    with tarfile.open(ROOT/'prepared/data.tar') as archive:
        matches = [m for m in archive.getmembers() if Path(m.name).name=='test.parquet']
        assert len(matches)==1 and matches[0].isfile()
        (local/'cache').mkdir()
        with archive.extractfile(matches[0]) as source, (local/'cache/test.parquet').open('wb') as target:
            shutil.copyfileobj(source,target)
    test_hash = dataset['parquet_files']['test.parquet']
    assert sha256(local/'cache/test.parquet')==test_hash
    assert pq.ParquetFile(local/'cache/test.parquet').metadata.num_rows==1919
    tokenizer = PreTrainedTokenizerFast.from_pretrained(local/'model',local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(local/'model',local_files_only=True)
    assert sum(p.numel() for p in model.parameters())==9967360
    with torch.inference_mode():
        encoded = tokenizer(['Add two numbers.'],return_tensors='pt',return_token_type_ids=False)
        assert model.generate(**encoded,max_new_tokens=4).shape[0]==1
    del model
    commit = (Path('source_commit.txt').read_text().strip() if Path('source_commit.txt').exists()
              else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (local/'source_commit.txt').write_text(commit+'\n')
    PREPARED.mkdir(parents=True)
    bundle = local/'inputs.tar'
    with tarfile.open(bundle,'w') as archive:
        for folder in ['model','cache']:
            archive.add(local/folder,arcname=folder)
        archive.add(local/'source_commit.txt',arcname='source_commit.txt')
        for source in sorted(Path('homeworks/hw2').glob('*.py')):
            archive.add(source,arcname=str(source))
    shutil.copyfile(bundle,PREPARED/'inputs.tar')
    digest = sha256(bundle)
    assert sha256(PREPARED/'inputs.tar')==digest
    ready = dict(source_commit=commit,inputs_sha256=digest,model_sha256=MODEL_HASH,
                 test_sha256=test_hash,cpu_preflight_passed=True,
                 model_files=selected,rows=1919,required_gpu='T4',
                 variants=[64,128,256],rounds=2,precision='fp32',
                 expected_outputs='exact model decodes in official row order',
                 budget_seconds=600,hard_process_timeout_seconds=720)
    (PREPARED/'ready.json').write_text(json.dumps(ready,indent=2))
    assert json.loads((PREPARED/'ready.json').read_text())==ready
    print('INFERENCE_CPU_READY',json.dumps(ready),flush=True)


def run_gpu():
    from google.colab import runtime
    try:
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert 'T4' in torch.cuda.get_device_name(), 'This probe requires the actual grading GPU type'
        drive_preflight(ROOT)
        ready = json.loads((PREPARED/'ready.json').read_text())
        assert ready['cpu_preflight_passed'] and ready['model_sha256']==MODEL_HASH
        assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
        assert sha256(Path('model/model.safetensors'))==MODEL_HASH
        assert sha256(Path('cache/test.parquet'))==ready['test_sha256']
        assert not RESULTS.exists(), 'Inspect previous results before repeating'
        log_path = PREPARED/'execution.log'
        with log_path.open('w') as log:
            completed = subprocess.run([sys.executable,'-u','-m','homeworks.hw2.benchmark_inference',
                '--model-dir','model','--cache-dir','cache','--output-dir',str(RESULTS),
                '--expected-model-sha256',MODEL_HASH,'--batches','64','128','256',
                '--budget-seconds','600','--profile'],stdout=log,stderr=subprocess.STDOUT,timeout=720)
        (PREPARED/'execution.complete.json').write_text(json.dumps({'returncode':completed.returncode}))
        assert completed.returncode==0
    except BaseException as error:
        (PREPARED/'execution.error.txt').write_text(type(error).__name__+': '+str(error))
        raise
    finally:
        try:
            runtime.unassign()
        except Exception as error:
            (PREPARED/'release-error.txt').write_text(str(error))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    run_gpu() if args.gpu else prepare()
