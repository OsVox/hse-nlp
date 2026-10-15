"""Verify a Drive-exported checkpoint's exact programs before bot submission."""
import argparse
import csv
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from .checkpoint_io import sha256

QUESTION_ORDER_SHA256 = '214746f8d4bc918b6fa1fde8ca3185a518a60637f239a2c45efa9a8143f612d3'


def verify(directory, tokenizer_directory, expected_commit):
    manifest = json.loads((directory/'checkpoint.complete.json').read_text())
    result = json.loads((directory/'results.complete.json').read_text())
    provenance = json.loads((directory/'predictions.provenance.json').read_text())
    assert directory.name == f"step-{result['step']:06d}"
    assert manifest['status'] == 'checkpoint_verified'
    for name, digest in manifest['files'].items():
        assert Path(name).name == name
        if (directory/name).exists():
            assert sha256(directory/name) == digest, name
    assert sha256(tokenizer_directory/'tokenizer.json') == manifest['files']['tokenizer.json']
    assert provenance['model_sha256'] == manifest['files']['model.safetensors']
    assert provenance['source_commit'] == expected_commit
    assert provenance['questions_sha256'] == QUESTION_ORDER_SHA256
    assert provenance['handwritten_repairs'] == 0
    assert sha256(directory/'predictions.csv') == provenance['output_sha256']
    assert sha256(directory/'predictions.jsonl') == result['jsonl_sha256']
    rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 1919 and all(set(row) == {'code'} for row in rows)
    programs = [row['code'] for row in rows]
    assert all(isinstance(code, str) and code.strip() for code in programs)
    with (directory/'predictions.csv').open(newline='', encoding='utf-8') as source:
        assert programs == [row['code'] for row in csv.DictReader(source)]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_directory, local_files_only=True)
    assert programs == tokenizer.batch_decode(provenance['selected_token_ids'],
        skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {'step': result['step'], 'rows': len(rows), 'jsonl_sha256': result['jsonl_sha256'],
            'model_sha256': provenance['model_sha256'], 'source_commit': expected_commit,
            'questions_sha256': QUESTION_ORDER_SHA256, 'status': 'exact_export_verified',
            'generation_seconds': provenance['generation_seconds'], 'gpu': provenance['gpu'],
            't4_timing_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Directory containing the checkpoint tokenizer files')
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.tokenizer, args.source_commit), indent=2))
