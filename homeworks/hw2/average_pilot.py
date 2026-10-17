"""CPU-only checkpoint averaging with predefined held-out selection gates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .checkpoint_io import sha256


def tokenizer_assets_match(left, right, allow_runtime_settings=False):
    if any(json.loads((left/name).read_text()) != json.loads((right/name).read_text())
           for name in ['tokenizer_config.json','special_tokens_map.json']):
        return False
    if sha256(left/'tokenizer.json') == sha256(right/'tokenizer.json'):
        return True
    if not allow_runtime_settings:
        return False
    # Fast-tokenizer calls explicitly override these two saved backend settings.
    # Vocabulary, token IDs, normalization and segmentation must still be equal.
    a=json.loads((left/'tokenizer.json').read_text())
    b=json.loads((right/'tokenizer.json').read_text())
    for record in (a,b):
        record.pop('padding',None)
        record.pop('truncation',None)
    return a == b


def average_checkpoints(parents, destination, allow_runtime_tokenizer_settings=False):
    from safetensors.torch import load_file, save_file
    import torch
    assert len(parents) >= 2 and not destination.exists()
    records, tensors = [], None
    base_config = json.loads((parents[-1]/'config.json').read_text())
    for parent in parents:
        manifest = json.loads((parent/'checkpoint.complete.json').read_text())
        assert manifest['status'] == 'checkpoint_verified'
        assert sha256(parent/'model.safetensors') == manifest['files']['model.safetensors']
        assert tokenizer_assets_match(parent, parents[-1],
                                      allow_runtime_settings=allow_runtime_tokenizer_settings)
        assert json.loads((parent/'config.json').read_text()) == base_config
        state = load_file(str(parent/'model.safetensors'), device='cpu')
        assert all(t.is_floating_point() and torch.isfinite(t).all() for t in state.values())
        if tensors is None:
            tensors = {k: t.float().clone() for k, t in state.items()}
        else:
            assert tensors.keys() == state.keys()
            for k, t in state.items():
                assert tensors[k].shape == t.shape
                tensors[k].add_(t.float())
        records.append({'path':str(parent), 'model_sha256':manifest['files']['model.safetensors'],
                        'tokenizer_sha256':sha256(parent/'tokenizer.json'),
                        'coefficient':1/len(parents)})
        del state
    for t in tensors.values():
        t.div_(len(parents))
    destination.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(destination/'model.safetensors'), metadata={'format':'pt'})
    for name in ['config.json','generation_config.json','tokenizer.json',
                 'tokenizer_config.json','special_tokens_map.json','step.json']:
        shutil.copyfile(parents[-1]/name, destination/name)
    (destination/'averaging.json').write_text(json.dumps({'method':'arithmetic_mean',
        'parents':records,'training_updates_added':0,'inference_only':True,
        'allow_runtime_tokenizer_settings':allow_runtime_tokenizer_settings}, indent=2))
    manifest = {'status':'checkpoint_verified', 'files':{
        p.name:sha256(p) for p in destination.iterdir() if p.is_file()}}
    (destination/'checkpoint.complete.json').write_text(json.dumps(manifest, indent=2))
    return records


def passes(candidate, baseline):
    assert candidate['usable_references'] == baseline['usable_references']
    return (candidate['numeric_correct'] > baseline['numeric_correct'] and
            candidate['prediction_failures'] <= baseline['prediction_failures'])


def run(root, output):
    import torch
    assert not torch.cuda.is_available(), 'This pilot must use CPU only'
    torch.set_num_threads(2)
    assert not output.exists(), 'Preserve previous experiments'
    output.mkdir(parents=True)
    parents = root/'runs/full-42m-100k-v1'
    plan = {'baseline':'step-070000','candidates':{'avg60_70':[60000,70000],
        'avg50_60_70':[50000,60000,70000]},'pilot_rows':[512,768],
        'confirmation_rows':[768,1024], 'num_beams':1, 'do_sample':False,
        'gate':'Strictly more numeric agreements than baseline and no more execution failures in both stages',
        'training_updates_added':0, 'outside_data':False,
        'source':'https://arxiv.org/html/1706.03762v7#S6.SS1',
        'note':'Diagnostic references are original held-out TRAIN rows; not public test answers.'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2))
    models = {'baseline':parents/'step-070000'}
    for name, steps in plan['candidates'].items():
        destination = output/'models'/name/'step-070000'
        average_checkpoints([parents/f'step-{s:06d}' for s in steps], destination)
        models[name] = destination
    def evaluate(name, stage, offset):
        result = output/stage/(name+'.json')
        result.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable,'-u','-m','homeworks.hw2.numeric_eval',
            '--model-dir',str(models[name]),'--cache-dir','/content/hw2_cache',
            '--output',str(result),'--offset',str(offset),'--rows','256',
            '--beams','1','--cpu-threads','2']
        with result.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=300)
        return json.loads(result.read_text())
    pilot = {name:evaluate(name, 'pilot', 512) for name in models}
    eligible = [name for name in plan['candidates'] if passes(pilot[name],pilot['baseline'])]
    selection = {'pilot':pilot,'selected':None,'confirmed':False}
    if eligible:
        winner = max(eligible, key=lambda name:pilot[name]['numeric_correct'])
        confirmation = {name:evaluate(name,'confirmation',768) for name in ['baseline',winner]}
        selection.update(selected=winner, confirmation=confirmation,
                         confirmed=passes(confirmation[winner],confirmation['baseline']))
    (output/'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    # Export the final trained baseline, and an averaged model only after it
    # passes both held-out gates. The public test never chooses the average.
    from .greedy_export import run as export
    for name in ['baseline'] + ([selection['selected']] if selection['confirmed'] else []):
        export(models[name], Path('/content/hw2_cache'),
               output/'exports'/name/'step-070000', batch_size=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
