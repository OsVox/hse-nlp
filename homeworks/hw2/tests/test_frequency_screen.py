from homeworks.hw2.frequency_screen import PLAN, select


def test_gate_rejects_split_tie_and_more_failures():
    def arm(a,b,f=1):
        return {'pilot':{'numeric_correct':a,'prediction_failures':f},
                'confirmation':{'numeric_correct':b,'prediction_failures':f}}
    r=select({'control':arm(10,20),'tie':arm(10,22),'failures':arm(12,23,2),'winner':arm(11,21)})
    assert r['shortlist']==['winner']
    assert all(not row['eligible'] for row in r['ranking'] if row['arm']!='winner')
    assert r['training_updates_per_arm']==1024 and not r['long_run_authorized_by_evidence']


def test_matched_plan_changes_only_sampling_power():
    arms=list(PLAN['arms'].values())
    assert {a['frequency_power'] for a in arms}=={0,.5,1}
    common=[{k:v for k,v in a.items() if k!='frequency_power'} for a in arms]
    assert common[0]==common[1]==common[2]
    assert common[0]['numeric_token_weight']==2
    assert PLAN['final_step']-PLAN['parent_step']==1024
    assert PLAN['training_updates_per_arm']%PLAN['group_window']==0
