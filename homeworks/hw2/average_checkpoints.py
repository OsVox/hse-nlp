"""Average verified checkpoints from our own training run on CPU.

This creates one inference model, without new examples, target programs or
optimizer state. It does not resume training or edit any parent checkpoint.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import T5ForConditionalGeneration

from .checkpoint_io import sha256

ASSETS = ('config.json', 'generation_config.json', 'tokenizer.json',
          'tokenizer_config.json', 'special_tokens_map.json')


def average(parents, output):
    if len(parents) < 2 or len({str(p.resolve()) for p, _ in parents}) != len(parents):
        raise ValueError('At least two distinct parent checkpoints are required')
    if output.exists():
        raise FileExistsError('Use a new output directory')
    records, tensors, asset_hashes = [], [], None
    for directory, expected_model_hash in parents:
        manifest = json.loads((directory/'checkpoint.complete.json').read_text())
        required = {'model.safetensors', 'run_config.json', 'step.json', *ASSETS}
        if manifest['status'] != 'checkpoint_verified' or not required.issubset(manifest['files']):
            raise ValueError('Parent is missing verified model assets')
        for name, expected in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe parent manifest filename')
            if name != 'training_state.pt' and sha256(directory/name) != expected:
                raise ValueError(f'Parent file hash mismatch: {name}')
        if manifest['files']['model.safetensors'] != expected_model_hash:
            raise ValueError('Parent does not match the selected checkpoint hash')
        current_assets = {name: manifest['files'][name] for name in ASSETS}
        if asset_hashes is not None and current_assets != asset_hashes:
            raise ValueError('Parents have different model or tokenizer settings')
        asset_hashes = current_assets
        records.append({'path': str(directory), 'model_sha256': expected_model_hash,
                        'manifest_sha256': sha256(directory/'checkpoint.complete.json'),
                        'run_config_sha256': manifest['files']['run_config.json'],
                        'step': json.loads((directory/'step.json').read_text())['step']})
        tensors.append(load_file(str(directory/'model.safetensors'), device='cpu'))
    # Identical run settings also bind the initialization/continuation history.
    if len({record['run_config_sha256'] for record in records}) != 1:
        raise ValueError('Only checkpoints from the same training run are supported')
    keys = set(tensors[0])
    if any(set(item) != keys for item in tensors[1:]):
        raise ValueError('Parent tensor names differ')
    combined = {}
    for name in sorted(keys):
        values = [item[name] for item in tensors]
        first = values[0]
        if any(v.shape != first.shape or v.dtype != first.dtype for v in values):
            raise ValueError(f'Incompatible tensor: {name}')
        if first.is_floating_point():
            if any(not torch.isfinite(v).all() for v in values):
                raise ValueError(f'Non-finite parent tensor: {name}')
            combined[name] = (sum(v.double() for v in values)/len(values)).to(first.dtype)
        else:
            if any(not torch.equal(first, v) for v in values[1:]):
                raise ValueError(f'Non-floating buffers differ: {name}')
            combined[name] = first.clone()
    output.mkdir(parents=True, exist_ok=False)
    for name in ASSETS:
        shutil.copyfile(parents[0][0]/name, output/name)
    save_file(combined, str(output/'model.safetensors'), metadata={'format': 'pt'})
    restored = load_file(str(output/'model.safetensors'), device='cpu')
    assert all(torch.equal(restored[name], value) for name, value in combined.items())
    model, loading = T5ForConditionalGeneration.from_pretrained(
        output, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(key) for key in
                   ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    record = {'method': 'uniform arithmetic weight average', 'parents': records,
              'weights': [1/len(parents)]*len(parents), 'device': 'cpu',
              'accumulation_dtype': 'float64', 'parameters': sum(p.numel() for p in model.parameters()),
              'script_sha256': sha256(Path(__file__)), 'resume_supported': False,
              'training_targets_created': False, 'model_sha256': sha256(output/'model.safetensors')}
    (output/'averaging.json').write_text(json.dumps(record, indent=2)+'\n')
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'checkpoint.complete.json').write_text(json.dumps(
        {'status': 'checkpoint_verified', 'files': files}, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', nargs=2, action='append', required=True,
                        metavar=('DIRECTORY', 'EXPECTED_MODEL_SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(average([(Path(p), h) for p, h in args.parent], args.output), indent=2))
