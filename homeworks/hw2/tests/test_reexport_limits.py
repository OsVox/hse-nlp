import csv
import json
import sys
from types import SimpleNamespace
import pytest
from homeworks.hw2 import reexport_limits as export
from homeworks.hw2.checkpoint_io import sha256


def test_fresh_generation_preserves_original_and_reuses_only_verified_output(tmp_path,monkeypatch):
    monkeypatch.setattr(export,'ROOT',tmp_path)
    source=tmp_path/'runs'/'trial'/'model'/'step-001921';source.mkdir(parents=True)
    (source/'model.safetensors').write_bytes(b'unchanged checkpoint')
    (source/'config.json').write_text('{}')
    (source/'predictions.jsonl').write_text('original oversize export stays untouched')
    manifest={'status':'checkpoint_verified','files':{n:sha256(source/n) for n in ['model.safetensors','config.json']}}
    (source/'checkpoint.complete.json').write_text(json.dumps(manifest))
    spec=dict(source=str(source),manifest_sha256=sha256(source/'checkpoint.complete.json'),
              model_sha256=sha256(source/'model.safetensors'),max_new_tokens=640)
    calls=[]
    def generate(model,output,cache,**kwargs):
        assert (model/'model.safetensors').read_bytes()==b'unchanged checkpoint'
        assert kwargs['max_new_tokens']==640 and kwargs['max_input_tokens']==0
        calls.append(kwargs)
        with output.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['code']);writer.writeheader()
            writer.writerows({'code':'def solve():\n    return 1'} for _ in range(1919))
        (output.parent/'predictions.provenance.json').write_text(json.dumps(dict(
            model_sha256=spec['model_sha256'],max_new_tokens=640,num_beams=1,syntax_retry_beams=0,
            do_sample=False,handwritten_repairs=0,max_input_tokens=None)))
    monkeypatch.setitem(sys.modules,'homeworks.hw2.inference',SimpleNamespace(generate_predictions=generate))
    export.regenerate(spec,tmp_path/'cache',local_root=tmp_path/'work')
    destination=source/'export-max640-v1'/source.name
    assert (source/'predictions.jsonl').read_text()=='original oversize export stays untouched'
    assert len((destination/'predictions.jsonl').read_text().splitlines())==1919
    export.regenerate(spec,tmp_path/'cache',local_root=tmp_path/'work')
    assert len(calls)==1
    (destination/'predictions.jsonl').write_text('corrupt')
    with pytest.raises(AssertionError):export.regenerate(spec,tmp_path/'cache',local_root=tmp_path/'work')
    assert len(calls)==1
