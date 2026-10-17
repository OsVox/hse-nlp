"""Collect a completed scale-up checkpoint from Drive without running the model.

This verifies files and prepares an exact bot attachment; it does not send it.
Weights stay outside the repository. The live GPU training process is untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .checkpoint_io import sha256
from .download_checkpoint import download
from .verify_coverage_checkpoint import read_verified, verify as verify_coverage
from .verify_export import verify as verify_export


def check_decoding(provenance):
    expected = dict(num_beams=1, syntax_retry_beams=0, do_sample=False,
        selection='greedy', max_new_tokens=1280, max_input_tokens=None,
        handwritten_repairs=0, precision='torch.float32', batch_size=64)
    for key, value in expected.items():
        if provenance.get(key) != value or key not in provenance:
            raise ValueError('Unexpected generation setting: '+key)


def collect(folder, destination, evidence, ready_path, parent, attachment, *, coverage_plan=None):
    if evidence.exists() or attachment.exists():
        raise ValueError('Inspect existing evidence/attachment before collecting again')
    download(folder, destination)
    listing = subprocess.check_output(['gdrive','files','list','--parent',folder,
        '--max','100','--skip-header','--full-name','--field-separator','|'],text=True)
    weights=[line.split('|')[0] for line in listing.splitlines()
             if line.split('|')[1:3]==['model.safetensors','regular']]
    if len(weights)!=1:
        raise ValueError('Missing or ambiguous model weights')
    subprocess.run(['gdrive','files','download',weights[0],
                    '--destination',str(destination)],check=True)
    manifest=json.loads((destination/'checkpoint.complete.json').read_text())
    if sha256(destination/'model.safetensors')!=manifest['files']['model.safetensors']:
        raise ValueError('Downloaded model hash mismatch')
    ready=json.loads(ready_path.read_text())
    provenance=json.loads((destination/'predictions.provenance.json').read_text())
    check_decoding(provenance)
    verification=verify_export(destination,destination,ready['source_commit'])
    plan=dict(ready['plan'] if coverage_plan is None else coverage_plan,
              eligible_rows=ready['fresh_preparation']['new_eligible_rows'])
    coverage=verify_coverage(read_verified(destination,'coverage.json'),
        read_verified(destination,'run_config.json'),plan,read_verified(parent,'coverage.json'))
    if coverage['step']!=verification['step']:
        raise ValueError('Coverage and generation steps differ')
    evidence.mkdir(parents=True)
    for p in destination.iterdir():
        if p.is_file() and p.suffix in {'.json','.jsonl','.csv'}:
            shutil.copyfile(p,evidence/p.name)
    for name,record in [('verification.json',verification),('coverage-verification.json',coverage)]:
        (evidence/name).write_text(json.dumps(record,indent=2)+'\n')
    attachment.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(destination/'predictions.jsonl',attachment)
    assert sha256(attachment)==verification['jsonl_sha256']
    record=dict(folder_id=folder,model_file_id=weights[0],downloaded_model_verified=True,
        attachment=str(attachment.resolve()),attachment_sha256=sha256(attachment),
        bot_submission_pending=True,step=verification['step'])
    (evidence/'collection.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folder',required=True)
    for key in ['destination','evidence','ready','parent','attachment']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(collect(a.folder,a.destination,a.evidence,a.ready,a.parent,a.attachment),indent=2))
