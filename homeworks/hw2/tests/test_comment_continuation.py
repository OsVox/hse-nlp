from homeworks.hw2.comment_continuation import PLAN, LAUNCH, select


def test_matched_budget_and_cleanup_only():
    assert PLAN['arms']['comments'] == dict(PLAN['arms']['control'], strip_comments=True)
    assert PLAN['final_step'] - PLAN['parent_step'] == 1024
    assert PLAN['final_step'] == PLAN['save_every'] == PLAN['eval_every']
    assert 1024 % PLAN['group_window'] == 0
    assert 'timeout=900' in LAUNCH and 'finalize_run(' in LAUNCH
    compile(LAUNCH, 'launch.py', 'exec')


def test_gate_requires_both_subset_gains_without_extra_failures():
    def arm(a, b, f=1):
        return {'pilot': {'numeric_correct': a, 'prediction_failures': f},
                'confirmation': {'numeric_correct': b, 'prediction_failures': f}}
    for candidate in (arm(10,22), arm(11,21,2), arm(9,25)):
        assert not select({'control': arm(10,20), 'comments':candidate})['shortlist']
    result=select({'control':arm(10,20),'comments':arm(11,21)})
    assert result['shortlist']==['comments']
    assert not result['long_run_authorized_by_evidence']
