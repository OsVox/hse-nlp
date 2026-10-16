"""CPU-only measurement of conservative comment/empty-line preprocessing."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .code_cleanup import strip_nonexecuting_text
from .numeric_eval import execute_numeric


def audit(shard, tokenizer_path, output, *, rows=4096, execution_pairs=256):
    began = time.monotonic()
    table = pq.read_table(shard, columns=['question', 'code'])
    heldout = set(table['question'].slice(0, 1024).to_pylist())
    eligible = [i for i, q in enumerate(table['question'].to_pylist()) if q not in heldout]
    indices = sorted(random.Random(2026101602).sample(eligible, min(rows, len(eligible))))
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.no_padding(); tok.no_truncation()
    counters = Counter(); lengths = [[], []]; pairs = []; evidence = []
    for index, row in zip(indices, table.take(indices).to_pylist()):
        original = row['code']
        cleaned, info = strip_nonexecuting_text(original)
        counters.update({k: int(v) for k, v in info.items()})
        counters['rows_with_comments'] += info['comments'] > 0
        try:
            tree = ast.parse(original)
            counters['rows_with_pass'] += any(isinstance(n, ast.Pass) for n in ast.walk(tree))
            counters['rows_with_docstrings'] += any(
                ast.get_docstring(n) is not None for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        except SyntaxError:
            pass
        for values, text in zip(lengths, [original, cleaned]):
            values.append(len(tok.encode(text).ids))
        if info['changed'] and len(pairs) < execution_pairs:
            pairs.append((original, cleaned))
        evidence.append(dict(row=index, original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                             cleaned_sha256=hashlib.sha256(cleaned.encode()).hexdigest(), **info))
    with ThreadPoolExecutor(max_workers=4) as pool:
        executions = list(pool.map(execute_numeric, [code for pair in pairs for code in pair]))
    check = Counter()
    for a, b in zip(executions[::2], executions[1::2]):
        check['both_numeric'] += a['status'] == b['status'] == 'ok'
        check['identical_numeric'] += a['status'] == b['status'] == 'ok' and a['value'] == b['value']
        check['status_mismatch'] += a['status'] != b['status']
        check['numeric_mismatch'] += a['status'] == b['status'] == 'ok' and a['value'] != b['value']
        check['unsupported_pairs'] += a['status'] != 'ok' or b['status'] != 'ok'
    result = dict(rows=len(indices), seed=2026101602, counters=dict(counters),
        shard_sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        original_mean_tokens=statistics.mean(lengths[0]), cleaned_mean_tokens=statistics.mean(lengths[1]),
        token_change_percent=100*(sum(lengths[1])/sum(lengths[0])-1),
        original_over_192=sum(n>192 for n in lengths[0]), cleaned_over_192=sum(n>192 for n in lengths[1]),
        execution_pairs=len(pairs), execution_checks=dict(check), elapsed_seconds=time.monotonic()-began,
        scope='Single original train shard; all exact copies of its first 1024 held-out question texts excluded. No test programs used.',
        preprocessing='One-to-one deletion of comments and empty lines outside strings; identical AST excluding locations required. No new examples or alternate solutions.',
        unchanged='Docstrings, pass statements, imports, variable names, assignments and all executable AST nodes retained.',
        limitations='Same AST does not preserve source/line-number introspection. Diagnostic execution is restricted and sampled; no accuracy or generation-speed gain is established. This does not prove all source targets correct.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'rows.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in evidence))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('CPU audit: no GPU needed')
    print(json.dumps(audit(args.shard, args.tokenizer, args.output), indent=2))
