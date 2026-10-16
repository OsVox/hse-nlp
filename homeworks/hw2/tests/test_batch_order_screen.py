from copy import deepcopy

from homeworks.hw2.batch_order_screen import PLAN, LAUNCH, select_pairs


def results():
    return {f'{arm}_s{seed}': {stage:dict(numeric_correct=100, prediction_failures=10)
                             for stage in ('pilot','confirmation')}
            for seed in (42,1042) for arm in ('control','shuffled')}


def test_no_confirmation_from_one_seed_or_from_increased_failures():
    r=results()
    r['shuffled_s42']['pilot']['numeric_correct']+=2
    assert not select_pairs(r)['confirmed']
    r['shuffled_s1042']['confirmation']['numeric_correct']+=1
    assert select_pairs(r)['confirmed']
    worse=deepcopy(r)
    worse['shuffled_s42']['pilot']['prediction_failures']+=1
    assert not select_pairs(worse)['confirmed']
    worse=deepcopy(r)
    worse['shuffled_s1042']['pilot']['numeric_correct']-=1
    assert not select_pairs(worse)['confirmed']


def test_budget_and_checkpoint_alignment():
    assert (PLAN['final_step']-PLAN['parent_step'])*PLAN['batch_size']==131072
    assert PLAN['training_updates_per_arm']%PLAN['group_window']==0
    assert all(PLAN['parent_step']<n<=PLAN['final_step'] for n in (PLAN['eval_every'],PLAN['save_every']))
    assert len(PLAN['arms'])==4
    assert 'timeout=1200' in LAUNCH
    assert 'homeworks.hw2.batch_order_screen' in LAUNCH
    compile(LAUNCH,'launch.py','exec')
