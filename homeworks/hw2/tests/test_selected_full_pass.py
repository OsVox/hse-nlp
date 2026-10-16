from copy import deepcopy
import pytest
from homeworks.hw2.learning_curve import OPTIONS, CONFIGS, select
from homeworks.hw2.learning_curve_confirm import paired_selection
from homeworks.hw2.selected_full_pass import build_plan, selected_recipe, verify_full_pass, launcher
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_unconfirmed_alternative_cannot_start_and_failed_repeat_falls_back():
    p = select(dict(adamw42=result(20,30),muon42=result(25,35),muon78=result(24,34)))
    with pytest.raises(ValueError): selected_recipe(p)
    failed = paired_selection(dict(adamw42=result(30,40),muon42=result(29,42)),p)
    assert selected_recipe(p,failed) == 'adamw42'
    passed = paired_selection(dict(adamw42=result(30,40),muon42=result(31,41)),p)
    assert selected_recipe(p,passed) == 'muon42'
    bad = deepcopy(p); bad['provisional_winner'] = 'muon78'
    with pytest.raises(ValueError): selected_recipe(bad,passed)


def test_full_pass_preserves_training_recipe_and_reuses_seed42():
    p = select(dict(adamw42=result(20,30),muon42=result(19,35),muon78=result(20,30)))
    config = dict(OPTIONS,**CONFIGS['adamw42'])
    budget = dict(optimizer_step=6018,completed_updates=6018,training_seconds=600.1,time_limit_reached=True)
    plan = build_plan(p,None,config,budget,11834223)
    assert plan['checkpoint_steps'] == [10000,20000,30000,40000,50000,60000,70000,80000,90000,92456]
    for key in ('steps','learning_rate','warmup','seed','optimizer','batch_size','untruncated_training'):
        assert plan['options'][key] == config[key]
    assert plan['options']['max_training_seconds'] is None
    assert plan['options']['checkpoint_training_seconds'] is None
    compile(launcher(plan),'launch.py','exec')
    assert f"timeout={plan['outer_timeout_seconds']}" in launcher(plan)
    bad = dict(config,seed=1042)
    with pytest.raises(ValueError): build_plan(p,None,bad,budget,11834223)


def test_completion_requires_every_row_once_and_full_sequences():
    r = dict(optimizer_step=5,state=dict(cursor=[1,0,0],untruncated_tokens=dict(question=100,code=200)),
        report=dict(start_step=0,completed_full_passes=1,eligible_rows=10,presentations=10,
        unique_rows_since_tracking_started=10,repeated_presentations_since_tracking_started=0,
        unique_visited_by_shard=[4,6],presentations_by_shard=[4,6],eligible_by_shard=[4,6],
        source_length=None,target_length=None,untruncated_tokens=dict(question=100,code=200)))
    plan = dict(final_step=5,eligible_rows=10)
    verify_full_pass(r,plan)
    for key,value in [('presentations',11),('unique_rows_since_tracking_started',9),
                      ('source_length',128),('target_length',192),('completed_full_passes',0)]:
        bad=deepcopy(r);bad['report'][key]=value
        with pytest.raises(AssertionError):verify_full_pass(bad,plan)
