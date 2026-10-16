import pytest
from homeworks.hw2.learning_curve import select, OPTIONS
from homeworks.hw2.learning_curve_confirm import build_plan,paired_selection,LAUNCH
from homeworks.hw2.tests.test_optimizer_pilot import result


def prior():
    return select(dict(adamw42=result(20,30),muon42=result(25,35),muon78=result(24,34)))


def test_confirmation_changes_only_seed_order_and_selected_arms():
    plan=build_plan(prior())
    assert list(plan['arms'])==['muon42','adamw42']
    assert plan['options']=={**OPTIONS,'seed':1042}
    assert plan['outer_timeout_seconds']==3160
    compile(LAUNCH,'launch.py','exec')
    assert 'timeout=3160' in LAUNCH and 'homeworks.hw2.learning_curve_confirm' in LAUNCH


def test_no_run_for_rejected_alternatives_or_inconsistent_selection():
    p=select(dict(adamw42=result(20,30),muon42=result(19,35),muon78=result(20,30)))
    with pytest.raises(ValueError):build_plan(p)
    p=prior();p['provisional_winner']='muon78'
    with pytest.raises(ValueError):build_plan(p)


def test_both_seeds_and_failure_counts_control_the_recipe():
    p=prior()
    r=paired_selection(dict(adamw42=result(30,40),muon42=result(31,41)),p)
    assert r['confirmed'] and r['full_pass_recipe']=='muon42'
    for candidate in [result(31,40),result(29,42),result(31,41,1)]:
        r=paired_selection(dict(adamw42=result(30,40),muon42=candidate),p)
        assert not r['confirmed'] and r['full_pass_recipe']=='adamw42'
        assert not r['public_test_used_for_selection']
    with pytest.raises(ValueError):paired_selection({'muon42':result(31,41)},p)
