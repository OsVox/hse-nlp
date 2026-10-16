from homeworks.hw2.tokenizer_confirmation import PLAN,LAUNCH,paired_selection


def arm(a,b,f=1):
    return {s:{'numeric_correct':v,'prediction_failures':f} for s,v in [('pilot',a),('confirmation',b)]}


def test_both_seeds_must_improve_both_subsets_without_extra_failures():
    good={'control':arm(10,20),'code8192':arm(11,22)}
    assert paired_selection(good,{'results':good})['confirmed']
    for bad in [dict(good,code8192=arm(10,22)),dict(good,code8192=arm(11,22,2)),dict(good,code8192=arm(9,24))]:
        assert not paired_selection(bad,{'results':good})['confirmed']
        assert not paired_selection(good,{'results':bad})['confirmed']


def test_short_second_seed_plan_keeps_matched_vocabulary_initialization():
    assert PLAN['seed']==1042 and PLAN['initialization_vocab_size']==16384
    assert set(PLAN['arms'])=={'control','code8192'}
    assert PLAN['steps']==PLAN['save_every']==PLAN['eval_every']==2048
    assert PLAN['steps']%PLAN['group_window']==0
    assert PLAN['outer_timeout_seconds']==900
    assert 'homeworks.hw2.tokenizer_confirmation' in LAUNCH
    compile(LAUNCH,'launcher','exec')
