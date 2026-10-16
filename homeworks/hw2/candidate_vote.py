"""Held-out pilot: select exact model programs by execution agreement.

This creates no training data and never consults reference answers during
selection. The restricted diagnostic executor is not the course bot sandbox.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from .numeric_eval import execute_numeric, score_programs


def select_indices(executions: list[dict]) -> dict[str, int]:
    """Compare top beam, first executable beam and strict numeric majority.

    Candidates remain in model rank order. Pairwise numeric tolerance is not
    transitive, so a majority must agree with every member of its group.
    Ties retain the earliest model-ranked candidate. If nothing executes,
    retain the top model output unchanged.
    """
    if not executions:
        raise ValueError('At least one model candidate is required')
    valid = [i for i, r in enumerate(executions)
             if r.get('status') == 'ok'
             and isinstance(r.get('value'), (int, float))
             and not isinstance(r['value'], bool) and math.isfinite(r['value'])]
    first = valid[0] if valid else 0
    groups: list[list[int]] = []
    for i in valid:
        for group in groups:
            if all(math.isclose(executions[i]['value'], executions[j]['value'],
                                rel_tol=1e-6, abs_tol=1e-6) for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    largest = max(groups, key=len, default=[])
    majority = largest[0] if len(largest) > len(executions) / 2 else first
    return {'top_beam': 0, 'first_executable': first, 'strict_majority': majority}


def run(model_dir: Path, cache_dir: Path, output: Path, *, offset=512,
        rows=256, beams=4, batch_size=32):
    import torch
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    from .data import validation_examples
    if beams < 2 or rows < 1 or offset < 512 or offset + rows > 1024:
        raise ValueError('Use 2+ beams and reserved rows 512:1024')
    if batch_size < 1 or output.exists():
        raise ValueError('Use a positive batch size and a new output directory')
    examples = validation_examples(cache_dir)[offset:offset + rows]
    if len(examples) != rows:
        raise ValueError('Missing held-out rows')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True).to(device).eval()
    candidates, token_ids = [], []
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            batch = tokenizer([r['question'] for r in examples[start:start + batch_size]],
                              truncation=True, max_length=128, padding=True,
                              return_tensors='pt', return_token_type_ids=False).to(device)
            generated = model.generate(**batch, max_new_tokens=192, num_beams=beams,
                                       num_return_sequences=beams, do_sample=False,
                                       use_cache=True)
            ids = generated.tolist()
            codes = tokenizer.batch_decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            for j in range(0, len(codes), beams):
                candidates.append(codes[j:j + beams])
                token_ids.append(ids[j:j + beams])
    if device == 'cuda':
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    assert len(candidates) == rows and all(len(c) == beams for c in candidates)
    # Cache exact duplicate programs only within this evaluation; no references.
    unique = list(dict.fromkeys(code for group in candidates for code in group))
    execution_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        execution = dict(zip(unique, pool.map(execute_numeric, unique)))
    selections = [select_indices([execution[c] for c in group]) for group in candidates]
    selection_seconds = time.perf_counter() - execution_started
    output.mkdir(parents=True, exist_ok=False)
    # References enter only now, after selections have been fixed.
    references = [r['code'] for r in examples]
    metrics = {}
    for policy in selections[0]:
        predictions = [group[chosen[policy]] for group, chosen in zip(candidates, selections)]
        tokens = [group[chosen[policy]] for group, chosen in zip(token_ids, selections)]
        assert predictions == tokenizer.batch_decode(tokens, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
        metrics[policy] = score_programs(predictions, references)
        metrics[policy]['changed_from_top'] = sum(chosen[policy] != 0 for chosen in selections)
        (output/f'{policy}.jsonl').write_text(''.join(
            json.dumps({'row': offset+i, 'code': code, 'selected_token_ids': tokens[i]},
                       ensure_ascii=False) + '\n' for i, code in enumerate(predictions)))
    records = [dict(row=offset+i, candidates=group, candidate_token_ids=token_ids[i],
                    execution=[execution[c] for c in group], selected=selections[i])
               for i, group in enumerate(candidates)]
    (output/'candidates.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n'
                                                 for r in records))
    bundled = Path('source_commit.txt')
    commit = (bundled.read_text().strip() if bundled.exists() else
              subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    result = dict(source_commit=commit, model_sha256=hashlib.sha256(
        (model_dir/'model.safetensors').read_bytes()).hexdigest(), offset=offset, rows=rows,
        beams=beams, batch_size=batch_size, device=device, precision='fp32',
        generation_seconds=generation_seconds, selection_seconds=selection_seconds,
        handwritten_repairs=0, training=False, reference_used_for_selection=False,
        policy='Strict majority of all candidates; otherwise first executable; rank breaks ties',
        questions_sha256=hashlib.sha256(json.dumps([r['question'] for r in examples],
                                                  ensure_ascii=False).encode()).hexdigest(),
        metrics=metrics, t4_timing_verified=False)
    result['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.glob('*.jsonl')}
    (output/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--offset', type=int, default=512)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--beams', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    run(**vars(parser.parse_args()))
