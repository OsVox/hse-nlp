from homeworks.hw2.optimizer_confirmation import PLAN, paired_selection
from homeworks.hw2.tests.test_optimizer_pilot import result


def prior():
    return dict(adamw_reference='control',shortlist=['muon_low'],
                results=dict(control=result(18,33,219),muon_low=result(22,39,188)))


def test_requires_positive_gains_on_both_subsets_in_both_seeds():
    assert paired_selection(dict(control=result(10,20),muon_low=result(11,21)),prior())['confirmed']
    assert not paired_selection(dict(control=result(10,20),muon_low=result(10,25)),prior())['confirmed']
    p=prior();p['results']['muon_low']=result(18,50)
    assert not paired_selection(dict(control=result(10,20),muon_low=result(11,21)),p)['confirmed']


def test_more_execution_failures_rejects_and_rates_stay_fixed():
    assert not paired_selection(dict(control=result(10,20),muon_low=result(11,21,1)),prior())['confirmed']
    assert PLAN['seed']==1042 and PLAN['steps']==2048 and len(PLAN['arms'])==2
    assert PLAN['arm_options']['muon_low']['muon_learning_rate']==.001
    assert PLAN['arm_options']['control']['learning_rate']==.0003
