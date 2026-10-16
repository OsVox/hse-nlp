from homeworks.hw2.depth_confirmation import PLAN, LAUNCH, paired_selection
from homeworks.hw2.tests.test_optimizer_pilot import result


def prior():
    return dict(shortlist=['decoder_heavy'],results=dict(control=result(23,41,177),decoder_heavy=result(25,51,103)))


def test_both_splits_and_seeds_must_improve_without_more_failures():
    assert paired_selection(dict(control=result(10,20),decoder_heavy=result(11,21)),prior())['confirmed']
    for candidate in [result(10,21),result(11,19),result(11,21,1)]:
        assert not paired_selection(dict(control=result(10,20),decoder_heavy=candidate),prior())['confirmed']
    p=prior();p['results']['decoder_heavy']=result(23,51)
    assert not paired_selection(dict(control=result(10,20),decoder_heavy=result(11,21)),p)['confirmed']


def test_plan_keeps_recipe_and_bounds_and_records_distinct_shapes():
    assert PLAN['seed']==1042 and PLAN['steps']==2048 and PLAN['batch_size']==128
    assert set(PLAN['arms'])=={'control','decoder_heavy'}
    assert not PLAN['require_identical_initialization']
    assert PLAN['arm_options']['decoder_heavy']['decoder_layers']==6
    assert PLAN['arm_options']['control']['decoder_layers']==4
    for v in PLAN['arm_options'].values():
        assert v['muon_learning_rate']==.001 and v['learning_rate']==.0003
    assert 'timeout=1100' in LAUNCH and 'RTX PRO 6000' in LAUNCH
    compile(LAUNCH,'launch.py','exec')
