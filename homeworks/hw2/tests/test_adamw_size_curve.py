from copy import deepcopy

import pytest

from homeworks.hw2.adamw_size_curve import PLAN, LAUNCH, select


def measurement(correct,failures=3):
    return {stage:dict(numeric_correct=correct,prediction_failures=failures,
        questions_sha256=stage,evaluation_examples_sha256='examples',evaluator_sha256='evaluator',
        offset=offset,rows=1024,usable_references=982 if offset==0 else 988,
        beams=1,device='cuda',precision='torch.float32',batch_size=64,
        max_input_tokens=None,max_new_tokens=1280,handwritten_repairs=0)
        for stage,offset in [('pilot',0),('confirmation',1024)]}


def test_gate_requires_two_gains_and_no_added_failures():
    reference=dict(results=measurement(10),model_sha256='parent')
    good=measurement(11)
    assert select({'adamw78':good},reference)['second_seed_required']
    assert select({'adamw78':measurement(10)},reference)['provisional_winner']=='adamw42'
    assert select({'adamw78':measurement(11,4)},reference)['provisional_winner']=='adamw42'
    good['confirmation']['numeric_correct']=9
    assert select({'adamw78':good},reference)['provisional_winner']=='adamw42'


@pytest.mark.parametrize('key,value',[
    ('questions_sha256','different'),('evaluator_sha256','changed'),
    ('max_input_tokens',128),('max_new_tokens',192),('precision','torch.bfloat16')])
def test_incomparable_reference_is_rejected(key,value):
    reference=dict(results=measurement(10),model_sha256='parent')
    candidate=deepcopy(measurement(11));candidate['pilot'][key]=value
    with pytest.raises(ValueError,match='Incomparable'):
        select({'adamw78':candidate},reference)


def test_only_one_bounded_adamw_arm_and_no_automatic_promotion():
    assert set(PLAN['arms'])=={'adamw78'}
    assert PLAN['arms']['adamw78']['optimizer']=='adamw'
    assert PLAN['options']['max_training_seconds']==600
    assert PLAN['options']['checkpoint_training_seconds']==300
    assert PLAN['options']['untruncated_training']
    assert not PLAN['long_run_started'] and not PLAN['public_test_used_for_selection']
    compile(LAUNCH,'launch.py','exec')
    assert 'timeout=2100' in LAUNCH and 'runtime.unassign' in LAUNCH
    assert 'homeworks.hw2.adamw_size_curve' in LAUNCH
