"""Bounded CPU screen of fixed adjacent checkpoint averages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .average_pilot import average_checkpoints
from .checkpoint_io import sha256


def run(root, output, cache_dir, phase):
    import torch
    assert not torch.cuda.is_available(), 'CPU-only experiment'
    torch.set_num_threads(4)
    plan = json.loads((output/'plan.json').read_text())
    if phase == 'build':
        for name, steps in plan['candidates'].items():
            average_checkpoints([root/'parents'/f'step-{s:06d}' for s in steps],
                                root/'models'/name,
                                allow_runtime_tokenizer_settings=plan.get('allow_runtime_tokenizer_settings',False))
        print('AVERAGES_BUILT', flush=True)
        return
    # Evaluate only the model archive restored and byte-verified from Drive.
    receipt = json.loads((output/'models-drive-verification.json').read_text())
    assert receipt['download_verified'] and receipt['restored_models_verified']
    examples = root/'questions.jsonl' if 'examples_file_sha256' in plan else None
    if examples is not None:
        assert sha256(examples) == plan['examples_file_sha256']
    deadline = time.monotonic()+plan.get('cpu_timeout_seconds',7200)
    models = {'baseline': root/'parents'/f"step-{plan['baseline']:06d}"}
    models.update({name:root/'restored-models'/name for name in plan['candidates']})
    def evaluate(name, stage):
        result = output/stage/f'{name}.json'
        assert not result.exists(), 'Preserve previous evaluations'
        result.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'homeworks.hw2.numeric_eval',
                   '--model-dir', str(models[name]), '--cache-dir', str(cache_dir),
                   '--output', str(result), '--offset', str(plan[stage]['offset']),
                   '--rows', str(plan[stage]['rows']), '--beams', '1', '--cpu-threads', '4',
                   '--max-input-tokens',str(plan.get('max_input_tokens',128)),
                   '--max-new-tokens',str(plan.get('max_new_tokens',192))]
        if examples is not None:
            command += ['--examples-file',str(examples)]
        manifest = json.loads((models[name]/'checkpoint.complete.json').read_text())
        assert manifest['status'] == 'checkpoint_verified'
        for asset, digest in manifest['files'].items():
            assert Path(asset).name == asset
            if asset != 'training_state.pt':
                assert sha256(models[name]/asset) == digest
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('CPU experiment budget exhausted')
        with result.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(1200,remaining))
        value = json.loads(result.read_text())
        assert value['device'] == 'cpu' and value['beams'] == 1
        print(stage, name, value['numeric_correct'], value['prediction_failures'], flush=True)
        return value
    pilot = {name:evaluate(name, 'pilot') for name in models}
    eligible = [name for name in plan['candidates']
                if pilot[name]['numeric_correct'] > pilot['baseline']['numeric_correct']]
    confirmation = {}
    confirmed = []
    if eligible:
        confirmation = {name:evaluate(name,'confirmation') for name in ['baseline']+eligible}
        for name in eligible:
            base_fail = sum(stage['baseline']['prediction_failures'] for stage in (pilot,confirmation))
            fail = sum(stage[name]['prediction_failures'] for stage in (pilot,confirmation))
            if confirmation[name]['numeric_correct'] > confirmation['baseline']['numeric_correct'] and fail <= base_fail:
                confirmed.append(name)
    selected = max(confirmed, key=lambda name:pilot[name]['numeric_correct']+confirmation[name]['numeric_correct']) if confirmed else None
    result = dict(pilot=pilot, confirmation=confirmation, selected=selected,
                  confirmed=confirmed, training_updates=0, public_test_used_for_selection=False,
                  inference_only=True, automatic_training_extension=False)
    (output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED',selected,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['build','evaluate'], required=True)
    run(**vars(parser.parse_args()))
