"""Compare generated and supplied programs on held-out TRAIN rows only.

Reference execution is for evaluation, never for creating training targets.
Programs run in separate limited processes with restricted imports/builtins.
This is a diagnostic runner, not the course bot's sandbox or public pass@1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

RUNNER = r'''
import ast, builtins, contextlib, io, json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
if sys.platform == 'linux':
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
allowed_modules = {'math', 'fractions', 'decimal', 'statistics', 'itertools',
                   'functools', 'collections', 'operator'}
def restricted_import(name, *args, **kwargs):
    if name not in allowed_modules:
        raise ValueError('unsupported import')
    return builtins.__import__(name, *args, **kwargs)
names = ['abs','all','any','bool','dict','divmod','enumerate','filter','float',
         'int','isinstance','len','list','map','max','min','pow','print','range',
         'reversed','round','set','sorted','str','sum','tuple','zip',
         'ValueError','ZeroDivisionError']
try:
    code = sys.stdin.read()
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('private name')
    env = {'__builtins__': {name: getattr(builtins, name) for name in names}}
    env['__builtins__']['__import__'] = restricted_import
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, '<heldout-program>', 'exec'), env)
        value = env['solve']()
    if isinstance(value, bool) or not isinstance(value, (int,float)):
        raise ValueError('not numeric')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('not finite')
    print(json.dumps({'status':'ok','value':value}))
except BaseException as error:
    print(json.dumps({'status':type(error).__name__}))
'''


def execute_numeric(code: str) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, '-I', '-c', RUNNER], input=code,
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode:
            return {'status': 'process_failure'}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except (ValueError, OSError):
        return {'status': 'runner_failure'}


def compare_programs(pair: tuple[str, str]) -> dict:
    predicted, reference = map(execute_numeric, pair)
    usable = reference['status'] == 'ok'
    correct = usable and predicted['status'] == 'ok' and math.isclose(
        predicted['value'], reference['value'], rel_tol=1e-6, abs_tol=1e-6,
    )
    return {'reference_status': reference['status'],
            'prediction_status': predicted['status'], 'correct': correct}


def score_programs(predictions: list[str], references: list[str]) -> dict:
    if not predictions or len(predictions) != len(references):
        raise ValueError('predictions and references must have equal nonzero length')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(compare_programs, zip(predictions, references)))
    usable = sum(r['reference_status'] == 'ok' for r in results)
    correct = sum(r['correct'] for r in results)
    return {
        'rows': len(results), 'usable_references': usable, 'numeric_correct': correct,
        'numeric_agreement_on_usable': correct / usable if usable else None,
        'prediction_failures': sum(r['prediction_status'] != 'ok' for r in results),
        'note': 'Agreement with supplied train references; not public pass@1. '
                'Incorrect references and restricted execution may affect this diagnostic.',
    }


def save_generation(output, predictions, references, metadata):
    """Persist an unchanged evaluation payload before a CPU scorer consumes it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_suffix('.predictions.jsonl')
    prediction_path.write_text(''.join(json.dumps({'row':metadata['offset']+i,'code':code},
                                               ensure_ascii=False)+'\n'
                                       for i,code in enumerate(predictions)))
    metadata = dict(metadata, predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest())
    reference_text = json.dumps(references, ensure_ascii=False)
    bundle = dict(version=1, metadata=metadata, references=references,
                  references_sha256=hashlib.sha256(reference_text.encode()).hexdigest())
    output.with_suffix('.generation.json').write_text(json.dumps(bundle,ensure_ascii=False))


def score_saved_generation(output):
    """CPU-only entry point: never imports torch or loads model weights."""
    began = time.monotonic()
    started_utc = time.time()
    bundle = json.loads(output.with_suffix('.generation.json').read_text())
    assert bundle['version'] == 1
    metadata = bundle['metadata']
    prediction_path = output.with_suffix('.predictions.jsonl')
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == metadata['predictions_sha256']
    references = bundle['references']
    assert hashlib.sha256(json.dumps(references,ensure_ascii=False).encode()).hexdigest() == bundle['references_sha256']
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert [r['row'] for r in rows] == list(range(metadata['offset'],metadata['offset']+len(references)))
    result = dict(score_programs([r['code'] for r in rows], references), **metadata)
    result['scoring_seconds'] = time.monotonic()-began
    result['scoring_started_utc'] = started_utc
    result['scoring_finished_utc'] = time.time()
    output.write_text(json.dumps(result,indent=2))
    return result


def load_examples(cache_dir, examples_file=None):
    if examples_file is None:
        from .data import validation_examples
        return validation_examples(cache_dir)
    rows = [json.loads(line) for line in examples_file.read_text().splitlines()]
    if not rows or [r.get('position') for r in rows] != list(range(len(rows))):
        raise ValueError('Explicit held-out positions must be contiguous and ordered')
    if any(not isinstance(r.get(k),str) or not r[k].strip()
           for r in rows for k in ('question','code')):
        raise ValueError('Explicit held-out examples must contain original question/code strings')
    if len({r['question'] for r in rows}) != len(rows):
        raise ValueError('Explicit held-out questions must be distinct groups')
    return rows


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/hw2_cache'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=256)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--beams', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=None)
    parser.add_argument('--examples-file', type=Path, default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--generate-only', action='store_true')
    modes.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if args.score_only:
        print(json.dumps(score_saved_generation(args.output),indent=2))
        return
    import torch
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError('CPU thread count must be positive')
        torch.set_num_threads(args.cpu_threads)
    examples = load_examples(args.cache_dir,args.examples_file)[args.offset:args.offset + args.rows]
    if len(examples) != args.rows:
        raise ValueError('requested rows exceed held-out split')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device).eval()
    predictions = []
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(examples), 64):
            batch = tokenizer([r['question'] for r in examples[start:start + 64]],
                              padding=True, truncation=True, max_length=128,
                              return_tensors='pt', return_token_type_ids=False).to(device)
            ids = model.generate(**batch, max_new_tokens=192, num_beams=args.beams,
                                 do_sample=False)
            predictions.extend(tokenizer.batch_decode(ids, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False))
    if device == 'cuda':
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    metadata = dict(model_dir=str(args.model_dir), offset=args.offset, beams=args.beams,
                  device=device, precision=str(next(model.parameters()).dtype),
                  cpu_threads=torch.get_num_threads(), generation_seconds=generation_seconds,
                  batch_size=64, max_input_tokens=128, max_new_tokens=192,
                  model_sha256=hashlib.sha256((args.model_dir/'model.safetensors').read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  questions_sha256=hashlib.sha256(json.dumps([r['question'] for r in examples],ensure_ascii=False).encode()).hexdigest(),
                  handwritten_repairs=0)
    if args.examples_file is not None:
        metadata['evaluation_examples_sha256'] = hashlib.sha256(args.examples_file.read_bytes()).hexdigest()
    save_generation(args.output,predictions,[r['code'] for r in examples],metadata)
    if args.generate_only:
        print('GENERATION_SAVED',str(args.output.with_suffix('.generation.json')))
    else:
        print(json.dumps(score_saved_generation(args.output), indent=2))


if __name__ == '__main__':
    main()
