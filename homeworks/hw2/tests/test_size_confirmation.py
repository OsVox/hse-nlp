from homeworks.hw2.size_confirmation import PLAN, LAUNCH, paired_selection
from homeworks.hw2.size_pilot import PLAN as PRIOR_PLAN
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_replication_rejects_split_regression_tie_and_extra_failures():
    prior=dict(shortlist=['small'],results=dict(control=result(22,37,219),small=result(27,49,117)))
    assert paired_selection(dict(control=result(10,20),small=result(11,21)),prior)['confirmed']
    for candidate in (result(10,21),result(11,19),result(11,21,1)):
        assert not paired_selection(dict(control=result(10,20),small=candidate),prior)['confirmed']
    prior['results']['small']=result(22,49)
    assert not paired_selection(dict(control=result(10,20),small=result(11,21)),prior)['confirmed']


def test_repeats_recipe_with_new_seed_and_reversed_arm_order():
    assert PLAN['seed']==1042 and list(PLAN['arms'])==['control','small']
    for name in PLAN['arms']:
        assert PLAN['arm_options'][name]==PRIOR_PLAN['arm_options'][name]
    for key in ('steps','batch_size','warmup','learning_rate','precision','mixed_rows_per_shard','initialization_vocab_size'):
        assert PLAN[key]==PRIOR_PLAN[key]
    assert PLAN['save_every']==PLAN['steps']==2048
    assert 'timeout=1100' in LAUNCH and 'RTX PRO 6000' in LAUNCH
    compile(LAUNCH,'launch.py','exec')
