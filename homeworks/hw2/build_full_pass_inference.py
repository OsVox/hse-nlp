"""Build a candidate notebook without replacing the retained best submission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_SHA = '66e7fc2bf6eceb680c2b5ce7b839dd1abbdde1c82fb1ff0d8aa43152a308bdb4'


def build(checkpoint, output):
    template = (ROOT/'hw2_inference_colab.ipynb').read_bytes()
    assert hashlib.sha256(template).hexdigest() == TEMPLATE_SHA, 'Review changed notebook template'
    receipt = json.loads((checkpoint/'bot_receipt.json').read_text())
    manifest = json.loads((checkpoint/'checkpoint.complete.json').read_text())
    provenance = json.loads((checkpoint/'predictions.provenance.json').read_text())
    ready = json.loads((checkpoint.parent/'ready.json').read_text())
    assert receipt['status'] == 'scored' and receipt['run'] == 'selected-full-pass-v2'
    assert receipt['model_sha256'] == manifest['files']['model.safetensors']
    assert receipt['source_commit'] == ready['source_commit'] == provenance['source_commit']
    expected = dict(batch_size=64, max_input_tokens=None, max_new_tokens=1280,
                    num_beams=1, syntax_retry_beams=0, do_sample=False,
                    precision='torch.float32', handwritten_repairs=0)
    assert all(provenance[k] == v for k,v in expected.items())
    n = json.loads(template)
    step = receipt['step']; name = f'step-{step:06d}'
    replacements = {
        'inference-42m-complete-curve-75776-greedy-v1': f'inference-full-pass-{step:06d}-greedy-v1',
        "prepared/complete-curve-4k-v1": 'prepared/selected-full-pass-v2',
        'runs/complete-curve-4k-v1/complete/step-075776': f'runs/selected-full-pass-v2/{name}',
        'd3c8136a68830bca45f26ab4fcad31450d344f81': receipt['source_commit'],
        '472cc28d5b1d877c50a4cabef771cd1c2fe27426787cf0ceb557f552d482204a': receipt['model_sha256'],
        'step-075776': name,
        "'pyarrow>=18'": "'pyarrow==23.0.1','tokenizers==0.22.2','numpy==2.1.3'",
        'from homeworks.hw2.greedy_export import run': 'from homeworks.hw2.inference import generate_predictions\n    from homeworks.hw2.checkpoint_io import package_jsonl',
    }
    for cell in n['cells']:
        s = ''.join(cell['source'])
        for old,new in replacements.items(): s = s.replace(old,new)
        cell['source'] = s.splitlines(keepends=True)
        if cell['cell_type'] == 'code': cell.update(outputs=[],execution_count=None)
    n['cells'][0]['source'] = [f'''# HW2 full-pass candidate: {step:,} updates\n\nThis candidate scored **{receipt['public_percent']:.2f}% ({receipt['public_correct']}/960)**. It does not replace the retained best model. The notebook is prepared and statically checked, but has not been executed.\n\nUses the exact frozen source, checkpoint hashes, full questions, output cap 1,280, batch 64, FP32 and strict greedy decoding from the scored export. CPU preparation precedes GPU use. Run in a separate runtime after training finishes. A rerun must be checked for exact output equality; different hardware or software can change generated outputs. T4 timing is unverified.\n''']
    s = ''.join(n['cells'][2]['source'])
    s = s.replace("assert digest(source/'source.tar') == source_ready['source_sha256']", "assert source_ready['source_sha256'] == "+repr(ready['source_sha256'])+"\nassert digest(source/'source.tar') == source_ready['source_sha256']")
    s = s.replace("test = fetch_parquet", "scored = json.loads((parent/'results.complete.json').read_text())\nassert scored['jsonl_sha256'] == "+repr(receipt['jsonl_sha256'])+"\nassert digest(parent/'predictions.jsonl') == scored['jsonl_sha256']\nshutil.copyfile(parent/'predictions.jsonl', model_dir/'scored-predictions.jsonl')\ntest = fetch_parquet")
    n['cells'][2]['source'] = s.splitlines(keepends=True)
    s = ''.join(n['cells'][4]['source'])
    old = '    run(model_dir,work/\'cache\',local,batch_size=256)'
    assert old in s
    s = s.replace(old, '''    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    assert not local.exists(), 'Preserve existing exports'
    count, seconds = generate_predictions(model_dir,local/'predictions.csv',work/'cache',
        batch_size=64,max_input_tokens=0,max_new_tokens=1280,
        num_beams=1,syntax_retry=False,precision='fp32')
    digest = package_jsonl(local/'predictions.csv',local/'predictions.jsonl')
    for filename in ['checkpoint.complete.json','step.json','config.json','generation_config.json',
                     'tokenizer.json','tokenizer_config.json','special_tokens_map.json','run_config.json']:
        shutil.copyfile(model_dir/filename,local/filename)
    result = dict(step=json.loads((model_dir/'step.json').read_text())['step'],rows=count,
        jsonl_sha256=digest,generation_seconds=seconds,source_commit=ready['source_commit'],
        selection='greedy',bot_status='pending',t4_timing_verified=False)
    (local/'results.complete.json').write_text(json.dumps(result,indent=2))''')
    s = s.replace("    gpu=provenance['gpu']", "    assert provenance['max_input_tokens'] is None and provenance['max_new_tokens']==1280\n    assert provenance['batch_size']==64 and provenance['precision']=='torch.float32'\n    evidence['matches_scored_programs']=(local/'predictions.jsonl').read_bytes()==(model_dir/'scored-predictions.jsonl').read_bytes()\n    evidence['runtime_versions']={name:__import__(name).__version__ for name in ['torch','transformers','tokenizers','numpy','pyarrow']}\n    gpu=provenance['gpu']")
    n['cells'][4]['source'] = s.splitlines(keepends=True)
    for i,cell in enumerate(n['cells']):
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'cell-{i}')
    assert output != ROOT/'hw2_inference_colab.ipynb'
    output.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    return dict(checkpoint_step=step,notebook_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                syntax_checked=True,executed=False,candidate_promoted=False,decoding=expected)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();print(json.dumps(build(args.checkpoint,args.output),indent=2))
