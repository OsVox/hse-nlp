"""Measure two exact full-test decoding passes on one actual T4; never train."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .checkpoint_io import package_jsonl, sha256
from .inference import generate_predictions


def run(model_dir, cache_dir, output_dir, expected_hash, *, beams=4, batch_size=128):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'T4' in torch.cuda.get_device_name()
    assert sha256(model_dir/'model.safetensors') == expected_hash
    assert sha256(cache_dir/'test.parquet') == '8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    step = json.loads((model_dir/'step.json').read_text())['step']
    record = dict(gpu=torch.cuda.get_device_name(), model_sha256=expected_hash,
                  num_beams=beams, batch_size=batch_size, precision='fp32', runs=[],
                  timing_scope='Full generation includes tokenization, sorting, decoding, syntax retry and local CSV writing. Model load and Drive copy excluded.',
                  completed=False, t4_timing_verified=False)
    try:
        warmup = tokenizer(['Add two numbers.']*8, return_tensors='pt',
                           return_token_type_ids=False).to('cuda')
        with torch.inference_mode():
            model.generate(**warmup, num_beams=beams, max_new_tokens=16, use_cache=True)
        torch.cuda.synchronize()
        record['load_and_warmup_seconds'] = time.monotonic()-started
        with tempfile.TemporaryDirectory(prefix='hw2-decoding-') as tmp:
            for repeat in range(2):
                local = Path(tmp)/f'run-{repeat}'/f'step-{step:06d}'
                local.mkdir(parents=True)
                rows, seconds = generate_predictions(model_dir, local/'predictions.csv', cache_dir,
                    model=model, tokenizer=tokenizer, batch_size=batch_size, num_beams=beams)
                assert rows == 1919
                digest = package_jsonl(local/'predictions.csv',local/'predictions.jsonl')
                for name in ('checkpoint.complete.json', 'config.json', 'generation_config.json',
                             'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                             'step.json', 'run_config.json', 'validation.json'):
                    path = model_dir/name
                    if path.exists():
                        shutil.copyfile(path,local/name)
                (local/'results.complete.json').write_text(json.dumps(dict(step=step,
                    jsonl_sha256=digest, bot_status='pending', public_pass_at_1=None,
                    question_order='official data/test/test.parquet', num_beams=beams)))
                destination = output_dir/f'run-{repeat}'/local.name
                destination.mkdir(parents=True)
                files = {}
                for path in local.iterdir():
                    target = destination/path.name
                    shutil.copyfile(path,target)
                    assert sha256(target) == sha256(path)
                    files[path.name] = sha256(target)
                entry = dict(repeat=repeat, rows=rows, generation_seconds=seconds,
                             jsonl_sha256=digest, files=files)
                record['runs'].append(entry)
                (output_dir/'results.json').write_text(json.dumps(record,indent=2))
                print('T4_DECODING_PASS',json.dumps(entry),flush=True)
                if seconds >= 180:
                    record['stopped_over_time_limit'] = True
                    break
        record['completed'] = len(record['runs']) == 2
        record['repeat_outputs_identical'] = (record['completed'] and
            len({r['jsonl_sha256'] for r in record['runs']}) == 1)
        record['t4_timing_verified'] = (record['repeat_outputs_identical'] and
            max(r['generation_seconds'] for r in record['runs']) < 180)
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        (output_dir/'results.json').write_text(json.dumps(record,indent=2))
        with zipfile.ZipFile(output_dir/'exports.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in output_dir.rglob('*'):
                if path.is_file() and path.suffix in {'.json','.jsonl','.csv'}:
                    archive.write(path,path.relative_to(output_dir))
    print('T4_DECODING_RESULT',json.dumps(record),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-model-sha256',required=True)
    parser.add_argument('--beams',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=128)
    args = parser.parse_args()
    run(args.model_dir,args.cache_dir,args.output_dir,args.expected_model_sha256,
        beams=args.beams,batch_size=args.batch_size)
