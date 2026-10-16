"""Score one decoding configuration; hardware timing is recorded, not gated."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import torch

from .checkpoint_io import package_jsonl, sha256
from .inference import generate_predictions


def run(model_dir, cache_dir, output_dir, expected_hash, *, beams=4, batch_size=64):
    assert sha256(model_dir/'model.safetensors') == expected_hash
    assert sha256(cache_dir/'test.parquet') == '8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    step = json.loads((model_dir/'step.json').read_text())['step']
    record = dict(model_sha256=expected_hash, num_beams=beams,
                  batch_size=batch_size, completed=False, training=False,
                  t4_timing_verified=False, timing_requirement='Deferred by user; quality first',
                  device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='hw2-quality-') as tmp:
            local = Path(tmp)/f'step-{step:06d}'
            local.mkdir()
            rows, seconds = generate_predictions(model_dir, local/'predictions.csv', cache_dir,
                                                batch_size=batch_size, num_beams=beams)
            assert rows == 1919
            digest = package_jsonl(local/'predictions.csv', local/'predictions.jsonl')
            for name in ('checkpoint.complete.json', 'config.json', 'generation_config.json',
                         'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                         'step.json', 'run_config.json', 'validation.json'):
                if (model_dir/name).exists():
                    shutil.copyfile(model_dir/name, local/name)
            (local/'results.complete.json').write_text(json.dumps(dict(step=step,
                jsonl_sha256=digest, bot_status='pending', public_pass_at_1=None,
                question_order='official data/test/test.parquet', num_beams=beams)))
            destination = output_dir/local.name
            destination.mkdir()
            files = {}
            for path in local.iterdir():
                target = destination/path.name
                shutil.copyfile(path, target)
                assert sha256(target) == sha256(path)
                files[path.name] = sha256(target)
            record.update(completed=True, rows=rows, generation_seconds=seconds,
                          jsonl_sha256=digest, files=files)
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        (output_dir/'results.json').write_text(json.dumps(record, indent=2))
        with zipfile.ZipFile(output_dir/'exports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in output_dir.rglob('*'):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.csv'}:
                    archive.write(path, path.relative_to(output_dir))
    print('DECODING_QUALITY_COMPLETE', json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--beams', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    run(args.model_dir, args.cache_dir, args.output_dir, args.expected_model_sha256,
        beams=args.beams, batch_size=args.batch_size)
