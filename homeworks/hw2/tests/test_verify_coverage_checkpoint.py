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


def second_pass_evidence():
    record,config,plan,parent=deepcopy(evidence(20000))
    final,_,_,_=evidence(92456)
    state,report=record['state'],record['report']
    config['steps']=plan['final_step']=184912
    record['optimizer_step']+=92456
    state['cursor'][0]=1
    state['presentations']+=final['state']['presentations']
    state['untruncated_tokens']={k:v+final['state']['untruncated_tokens'][k]
                               for k,v in state['untruncated_tokens'].items()}
    report.update(presentations=state['presentations'],completed_full_passes=1,
        unique_rows_since_tracking_started=plan['eligible_rows'],
        repeated_presentations_since_tracking_started=2559950,
        unique_visited_by_shard=report['eligible_by_shard'],
        presentations_by_shard=[n+a for n,a in zip(report['eligible_by_shard'],state['counts'])],
        untruncated_tokens=state['untruncated_tokens'])
    return record,config,plan,parent


def test_second_pass_counts_intentional_repeats_and_partial_batches():
    result=verify(*second_pass_evidence())
    assert result['step']==112456 and result['unique_rows']==11834223
    assert result['repeats']==2559950 and result['completed_full_passes']==1
    assert result['planned_passes']==2 and not result['planned_passes_complete']
    assert result['partial_batch_shortfall']==195


def test_completed_second_pass_has_every_row_twice():
    record,config,plan,parent=deepcopy(evidence(92456))
    state,report=record['state'],record['report']
    config['steps']=plan['final_step']=record['optimizer_step']=184912
    state['cursor']=[2,0,0]
    state['presentations']*=2
    state['untruncated_tokens']={k:2*v for k,v in state['untruncated_tokens'].items()}
    report.update(presentations=state['presentations'],completed_full_passes=2,
        repeated_presentations_since_tracking_started=plan['eligible_rows'],
        presentations_by_shard=[2*n for n in report['eligible_by_shard']],
        untruncated_tokens=state['untruncated_tokens'])
    result=verify(record,config,plan,parent)
    assert result['planned_passes_complete']
    assert result['unique_rows']==result['repeats']==11834223
    assert result['presentations']==23668446


@pytest.mark.parametrize('mutation',[
    lambda r:r['report'].__setitem__('repeated_presentations_since_tracking_started',0),
    lambda r:r['state'].__setitem__('cursor',[2,10,0]),
    lambda r:r['report']['presentations_by_shard'].__setitem__(0,0),
])
def test_second_pass_rejects_missing_repeats_and_invalid_cursor(mutation):
    record,config,plan,parent=second_pass_evidence();mutation(record)
    with pytest.raises(ValueError):verify(record,config,plan,parent)
