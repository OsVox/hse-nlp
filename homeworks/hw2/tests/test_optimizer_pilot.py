from homeworks.hw2.optimizer_pilot import PLAN, select


def result(a,b,failures=0):
    return {'pilot':dict(numeric_correct=a,prediction_failures=failures),
            'confirmation':dict(numeric_correct=b,prediction_failures=0)}


def test_muon_must_beat_best_adamw_not_only_low_rate():
    r=select(dict(control=result(10,20),adamw_high=result(15,25),
                  muon_low=result(12,24),muon_high=result(16,26)))
    assert r['adamw_reference']=='adamw_high'
    assert r['shortlist']==['muon_high']


def test_tie_or_more_failures_cannot_pass():
    r=select(dict(control=result(10,20),adamw_high=result(10,20),
                  muon_low=result(10,30),muon_high=result(20,30,1)))
    assert r['adamw_reference']=='control' and not r['shortlist']
    assert len(PLAN['arms'])==4 and PLAN['steps']==2048
