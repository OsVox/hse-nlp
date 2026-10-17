"""Reproduce the retained scored checkpoint on CPU using its frozen source.

This is an export reproducibility check, not a new training experiment or a
Colab notebook/T4 timing verification. Production files are read only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ['source', 'model', 'cache', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    signal.alarm(3600)
    started = time.time()
    commit = 'fdd3fe7412d5e916007e184766a55998162d3b52'
    assert (a.source/'source_commit.txt').read_text().strip() == commit
    manifest = json.loads((a.model/'checkpoint.complete.json').read_text())
    for name, expected in manifest['files'].items():
        assert Path(name).name == name
        if name != 'training_state.pt':
            assert digest(a.model/name) == expected, name
    assert digest(a.model/'model.safetensors') == 'c758cd0d3ba81806098867b63fcae44789b0e1ad8e9d8b8a3dcecb9f50a27441'
    scored = a.model/'predictions.jsonl'
    assert digest(scored) == '20498c920ddabf65b1c3510490ae2d1d08c0439f4c71b69742d7b4e1305162a5'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.chdir(a.source)
    sys.path.insert(0, str(a.source))
    import torch
    assert not torch.cuda.is_available(), 'This control is CPU only'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from homeworks.hw2.inference import generate_predictions
    from homeworks.hw2.checkpoint_io import package_jsonl
    import pyarrow.parquet as pq
    questions = pq.read_table(a.cache/'test.parquet', columns=['question'])['question'].to_pylist()
    assert len(questions) == 1919
    assert hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest() == '214746f8d4bc918b6fa1fde8ca3185a518a60637f239a2c45efa9a8143f612d3'
    print('CPU_REPLAY_STARTED', os.getpid(), flush=True)
    count, seconds = generate_predictions(a.model, a.output/'predictions.csv', a.cache,
        batch_size=64, max_input_tokens=0, max_new_tokens=1280,
        num_beams=1, syntax_retry=False, precision='fp32')
    actual_digest = package_jsonl(a.output/'predictions.csv', a.output/'predictions.jsonl')
    original = [json.loads(line)['code'] for line in scored.read_text().splitlines()]
    actual = [json.loads(line)['code'] for line in (a.output/'predictions.jsonl').read_text().splitlines()]
    assert count == len(actual) == len(original) == 1919
    mismatches = [i for i, (x, y) in enumerate(zip(original, actual)) if x != y]
    record = dict(status='completed', rows=count, source_commit=commit,
        model_sha256=digest(a.model/'model.safetensors'),
        scored_jsonl_sha256=digest(scored), replay_jsonl_sha256=actual_digest,
        identical_programs=count-len(mismatches), differing_row_indices=mismatches,
        byte_identical=actual_digest == digest(scored), generation_seconds=seconds,
        started_utc=started, finished_utc=time.time(), cpu_threads=4,
        runtime_versions={n: __import__(n).__version__ for n in
                          ['torch', 'transformers', 'tokenizers', 'numpy', 'pyarrow']},
        notebook_end_to_end_verified=False, t4_timing_verified=False,
        training=False, handwritten_repairs=0)
    (a.output/'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
