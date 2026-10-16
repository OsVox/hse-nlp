import json
from homeworks.hw2 import muon_transfer


def test_transfer_gate_uses_stronger_control_and_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(muon_transfer,'PREPARED',tmp_path)
    def stages(a,b,fail=5):
        return {k:dict(numeric_correct=v,prediction_failures=fail)
                for k,v in [('pilot',a),('confirmation',b)]}
    (tmp_path/'ready.json').write_text(json.dumps({'parent_heldout':stages(139,286)}))
    results={'control':stages(140,286),'adamw_reset':stages(141,287),
             'muon_low':stages(142,288),'muon_high':stages(142,285)}
    chosen=muon_transfer.select(results)
    assert chosen['adamw_reference']=='adamw_reset'
    assert chosen['shortlist']==['muon_low']
    results['muon_low']=stages(142,288,6)
    assert not muon_transfer.select(results)['shortlist']


def test_transfer_budget_and_arms_are_explicit():
    p=muon_transfer.PLAN
    assert p['final_step']-p['parent_step']==1024
    assert len(p['arms'])==4
    assert all(a['numeric_token_weight']==2 for a in p['arms'].values())
    assert p['arms']['adamw_reset']['reset_optimizer']
    assert all(p['arms'][a]['reset_optimizer'] for a in ['muon_low','muon_high'])
    compile(muon_transfer.LAUNCH,'launch.py','exec')
