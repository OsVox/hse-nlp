from copy import deepcopy
import json
from pathlib import Path
import pytest
from homeworks.hw2.verify_coverage_checkpoint import verify

ROOT=Path(__file__).resolve().parents[1]/'experiments'


def evidence(step):
    cp=ROOT/'selected-full-pass-v2'/f'step-{step:06d}'
    read=lambda p:json.loads(p.read_text())
    return (read(cp/'coverage.json'),read(cp/'run_config.json'),
            read(ROOT/'selected-full-pass-v2/ready.json')['plan'],
            read(ROOT/'matched-curve-v1/adamw42/step-006018/coverage.json'))


@pytest.mark.parametrize('step,rows,shortfall',[(10000,1280000,0),(20000,2559950,50)])
def test_real_checkpoint_counts_include_partial_batches(step,rows,shortfall):
    result=verify(*evidence(step))
    assert result['step']==step and result['unique_rows']==rows
    assert result['partial_batch_shortfall']==shortfall and not result['complete_pass']


@pytest.mark.parametrize('mutation',[
    lambda r:r['state'].__setitem__('identity','changed'),
    lambda r:r['state'].__setitem__('cursor',[0,10,824]),
    lambda r:r['report'].__setitem__('unique_rows_since_tracking_started',2560000),
    lambda r:r['report'].__setitem__('repeated_presentations_since_tracking_started',1),
    lambda r:r['report'].__setitem__('target_length',192),
    lambda r:r['state']['counts'].__setitem__(0,0),
    lambda r:r['state']['untruncated_tokens'].__setitem__('code',0),
])
def test_rejects_changed_or_inconsistent_evidence(mutation):
    record,config,plan,parent=deepcopy(evidence(20000));mutation(record)
    with pytest.raises(ValueError):verify(record,config,plan,parent)
