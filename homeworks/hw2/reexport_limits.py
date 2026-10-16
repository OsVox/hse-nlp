"""Regenerate an oversize submission with a declared smaller decoding limit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .checkpoint_io import sha256, package_jsonl
from .short_pilots import ROOT, copy_verified


def run_pending_exports(ready_path, cache):
    ready=json.loads(Path(ready_path).read_text())
    for spec in ready.get('pending_reexports', []):
        subprocess.run([sys.executable,'-u','-m','homeworks.hw2.reexport_limits',
            '--spec',json.dumps(spec),'--cache-dir',str(cache)],check=True,timeout=360)


def regenerate(spec, cache, *, local_root=Path('/content/hw2-reexports')):
    source=Path(spec['source'])
    assert source.resolve().is_relative_to((ROOT/'runs').resolve())
    assert sha256(source/'checkpoint.complete.json')==spec['manifest_sha256']
    manifest=json.loads((source/'checkpoint.complete.json').read_text())
    assert manifest['files']['model.safetensors']==spec['model_sha256']
    cap=spec['max_new_tokens'];assert isinstance(cap,int) and 0<cap<1280
    destination=source/f'export-max{cap}-v1'/source.name
    result_path=destination/'results.complete.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        provenance=json.loads((destination/'predictions.provenance.json').read_text())
        assert sha256(destination/'predictions.jsonl')==result['jsonl_sha256']
        assert provenance['model_sha256']==spec['model_sha256'] and provenance['max_new_tokens']==cap
        assert (destination/'predictions.jsonl').stat().st_size<=4_000_000
        print('REEXPORT_ALREADY_VERIFIED',destination,flush=True)
        return
    local=local_root/source.parent.name/source.name
    local.mkdir(parents=True,exist_ok=False);destination.mkdir(parents=True,exist_ok=False)
    for name,digest in manifest['files'].items():
        if name!='model.safetensors' and not name.endswith('.json'):continue
        assert Path(name).name==name and sha256(source/name)==digest
        copy_verified(source/name,local/name)
        if name.endswith('.json'):copy_verified(source/name,destination/name)
    copy_verified(source/'checkpoint.complete.json',destination/'checkpoint.complete.json')
    from .inference import generate_predictions
    generate_predictions(local,destination/'predictions.csv',cache,precision='fp32',
        batch_size=64,max_input_tokens=0,max_new_tokens=cap)
    digest=package_jsonl(destination/'predictions.csv',destination/'predictions.jsonl')
    assert (destination/'predictions.jsonl').stat().st_size<=4_000_000
    provenance=json.loads((destination/'predictions.provenance.json').read_text())
    assert provenance['model_sha256']==spec['model_sha256'] and provenance['max_new_tokens']==cap
    assert provenance['num_beams']==1 and not provenance['syntax_retry_beams'] and not provenance['do_sample']
    assert provenance['handwritten_repairs']==0 and provenance['max_input_tokens'] is None
    result=dict(step=int(source.name.removeprefix('step-')),jsonl_sha256=digest,
        original_checkpoint=str(source),max_new_tokens=cap,bot_status='pending',public_pass_at_1=None,
        question_order='official data/test/test.parquet',training=False,
        reason='Original model output exceeded the bot file-size limit; fresh greedy generation, no program edits.')
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print('REEXPORT_READY_FOR_BOT',destination,digest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--cache-dir',type=Path,required=True)
    args=p.parse_args();regenerate(json.loads(args.spec),args.cache_dir)
