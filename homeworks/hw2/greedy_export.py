"""Export one checkpoint using strictly greedy decoding and exact provenance."""
import argparse
import json
from pathlib import Path
import shutil

from .checkpoint_io import package_jsonl, sha256
from .inference import generate_predictions
from .verify_export import verify


def run(model_dir, cache_dir, output_dir, batch_size=64):
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    assert not output_dir.exists(), 'Preserve existing exports'
    manifest = json.loads((model_dir/'checkpoint.complete.json').read_text())
    assert sha256(model_dir/'model.safetensors') == manifest['files']['model.safetensors']
    count, seconds = generate_predictions(model_dir, output_dir/'predictions.csv', cache_dir,
        batch_size=batch_size, num_beams=1, syntax_retry=False, precision='fp32')
    digest = package_jsonl(output_dir/'predictions.csv', output_dir/'predictions.jsonl')
    for name in ['checkpoint.complete.json', 'step.json', 'config.json',
                 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'validation.json', 'run_config.json']:
        if (model_dir/name).exists():
            shutil.copyfile(model_dir/name, output_dir/name)
    provenance = json.loads((output_dir/'predictions.provenance.json').read_text())
    assert provenance['num_beams'] == 1 and provenance['syntax_retry_beams'] == 0
    result = dict(step=json.loads((model_dir/'step.json').read_text())['step'],
        rows=count, jsonl_sha256=digest, generation_seconds=seconds,
        source_commit=provenance['source_commit'], selection='greedy',
        bot_status='pending', t4_timing_verified=False)
    (output_dir/'results.complete.json').write_text(json.dumps(result, indent=2))
    evidence = verify(output_dir, output_dir, provenance['source_commit'])
    (output_dir/'export-verification.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    run(**vars(parser.parse_args()))
