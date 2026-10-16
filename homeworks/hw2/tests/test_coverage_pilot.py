import json
import pytest
from homeworks.hw2 import coverage_pilot as pilot
from homeworks.hw2.broad_screen import training_command


def test_new_sampler_removes_old_flag_and_keeps_checkpoint_exports():
    command=training_command(pilot.PLAN['arms']['coverage'])
    assert '--mixed-rows-per-shard' not in command and 'None' not in command
    assert '--coverage-index' in command
    assert '--predict-at-checkpoint' in command
    assert pilot.PLAN['final_step']-pilot.PLAN['parent_step']==1024
    assert len(pilot.PLAN['arms'])==2
    compile(pilot.LAUNCH,'launch.py','exec')


def test_quality_screen_requires_actual_coverage_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'RUN',tmp_path)
    checkpoint=tmp_path/'coverage'/f"step-{pilot.PLAN['final_step']:06d}"
    checkpoint.mkdir(parents=True)
    receipt=dict(optimizer_step=74048,report=dict(presentations=131072,
        unique_rows_since_tracking_started=131072,repeated_presentations_since_tracking_started=0,
        completed_full_passes=0,start_step=73024))
    (checkpoint/'coverage.json').write_text(json.dumps(receipt))
    def stages(a,b,fail):
        return {k:dict(numeric_correct=n,prediction_failures=fail)
                for k,n in [('pilot',a),('confirmation',b)]}
    results={'control':stages(139,286,20),'coverage':stages(138,285,20)}
    assert not pilot.select(results)['material_quality_regression']
    results['coverage']=stages(130,285,20)
    assert pilot.select(results)['material_quality_regression']
    receipt['report']['presentations']=100
    (checkpoint/'coverage.json').write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        pilot.select(results)
