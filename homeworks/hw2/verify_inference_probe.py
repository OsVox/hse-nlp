"""Verify every exact program in a Drive-exported inference comparison."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from transformers import PreTrainedTokenizerFast

from .checkpoint_io import sha256
from .verify_export import QUESTION_ORDER_SHA256


def verify(root, tokenizer_directory):
    plan = json.loads((root/'plan.json').read_text())
    assert plan['cpu_verified'] and plan['required_gpu'] == 'T4'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_directory, local_files_only=True)
    verified = []
    for item in plan['models']:
        folder = root/item['name']
        result = json.loads((folder/'results.json').read_text())
        ready = item['ready']
        assert result['completed'] and 'T4' in result['gpu']
        assert result['model_sha256'] == ready['model_sha256']
        assert result['test_sha256'] == ready['test_sha256']
        assert sha256(tokenizer_directory/'tokenizer.json') == ready['model_files']['tokenizer.json']
        baseline = None
        assert len(result['runs']) == 6
        for run in result['runs']:
            assert run['status'] == 'complete' and run['rows'] == 1919
            for name, digest in run['files'].items():
                assert Path(name).name == name and sha256(folder/name) == digest
            prefix = f"{result['precision']}-b{run['batch_size']}-r{run['repeat']}"
            output = folder/(prefix+'.jsonl')
            assert sha256(output) == run['jsonl_sha256']
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            assert len(rows) == 1919 and all(set(row)=={'code'} for row in rows)
            programs = [row['code'] for row in rows]
            assert all(isinstance(code,str) and code.strip() for code in programs)
            with (folder/(prefix+'.csv')).open(newline='') as source:
                assert programs == [row['code'] for row in csv.DictReader(source)]
            provenance = json.loads((folder/(prefix+'.provenance.json')).read_text())
            assert provenance['model_sha256'] == ready['model_sha256']
            assert provenance['source_commit'] == ready['source_commit']
            assert provenance['questions_sha256'] == QUESTION_ORDER_SHA256
            assert provenance['handwritten_repairs'] == 0
            assert provenance['gpu'] == result['gpu']
            assert provenance['generation_seconds'] == run['generation_seconds']
            assert provenance['output_sha256'] == sha256(folder/(prefix+'.csv'))
            assert programs == tokenizer.batch_decode(provenance['selected_token_ids'],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if baseline is None:
                baseline = programs
            assert run['changed_from_baseline'] == sum(a!=b for a,b in zip(programs,baseline))
        for summary in result['summary']:
            runs = [r for r in result['runs'] if r['batch_size']==summary['batch_size']]
            assert len(runs)==2 and {r['repeat'] for r in runs}=={0,1}
            assert summary['median_seconds'] == statistics.median(r['generation_seconds'] for r in runs)
            assert summary['max_seconds'] == max(r['generation_seconds'] for r in runs)
            assert summary['repeat_outputs_identical'] == (runs[0]['jsonl_sha256']==runs[1]['jsonl_sha256'])
            assert summary['all_outputs_equal_baseline'] == all(r['changed_from_baseline']==0 for r in runs)
        chosen = min(result['summary'], key=lambda s:s['median_seconds'])
        assert chosen['repeat_outputs_identical']
        verified.append(dict(name=item['name'], model_sha256=ready['model_sha256'],
            source_commit=ready['source_commit'], selected_batch=chosen['batch_size'],
            median_seconds=chosen['median_seconds'], max_seconds=chosen['max_seconds'],
            all_outputs_equal_baseline=chosen['all_outputs_equal_baseline'],
            t4_within_180_seconds=chosen['max_seconds']<=180,
            jsonl_sha256=next(r['jsonl_sha256'] for r in result['runs'] if r['batch_size']==chosen['batch_size']),
            rows=1919, status='every_export_and_token_decode_verified'))
    return verified


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.directory,args.tokenizer),indent=2))
