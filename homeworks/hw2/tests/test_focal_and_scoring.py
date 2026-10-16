"""Synthetic numerical/process tests, never additional training examples."""
import json
import subprocess
import sys

import pytest
import torch

from homeworks.hw2.async_scoring import ScoringJobs
from homeworks.hw2.numeric_eval import save_generation, score_saved_generation, score_programs
from homeworks.hw2.token_weighting import weighted_loss


@pytest.mark.parametrize('gamma',[0.,.5,1.,2.])
def test_focal_value_gradient_and_ignored_padding(gamma):
    torch.manual_seed(6)
    logits=torch.randn(2,3,5,requires_grad=True)
    labels=torch.tensor([[1,2,-100],[3,4,0]])
    actual=weighted_loss(logits,labels,torch.ones(5),focal_gamma=gamma)
    valid=labels.ne(-100)
    logp=logits.log_softmax(-1)[valid].gather(1,labels[valid][:,None]).flatten()
    expected=(-(1-logp.exp()).pow(gamma)*logp).mean()
    torch.testing.assert_close(actual,expected)
    grad=torch.autograd.grad(actual,logits,retain_graph=True)[0]
    torch.testing.assert_close(grad,torch.autograd.grad(expected,logits)[0])
    assert grad[~valid].count_nonzero()==0
    with pytest.raises(ValueError):
        weighted_loss(logits,labels,torch.ones(5),focal_gamma=-1)


def test_confident_and_padding_tokens_have_finite_focal_gradients():
    logits=torch.tensor([[[10000.,-10000.],[0.,0.]]],requires_grad=True)
    loss=weighted_loss(logits,torch.tensor([[0,-100]]),torch.ones(2),focal_gamma=.5)
    loss.backward()
    assert loss==0 and torch.isfinite(logits.grad).all()


def test_focal_screen_budget_and_gate():
    from homeworks.hw2.focal_screen import PLAN, LAUNCH, select
    assert PLAN['overlap_cpu_scoring'] and PLAN['training_updates_per_arm']==1024
    assert PLAN['final_step']-PLAN['parent_step']==1024
    assert PLAN['save_every']==PLAN['eval_every']==PLAN['final_step']
    def arm(a,b):return {s:{'numeric_correct':v,'prediction_failures':1} for s,v in [('pilot',a),('confirmation',b)]}
    assert select({'control':arm(10,20),'focal1':arm(11,21),'focal2':arm(10,22)})['shortlist']==['focal1']
    compile(LAUNCH,'launch.py','exec')


def test_saved_scoring_matches_direct_and_rejects_tampering(tmp_path):
    output=tmp_path/'pilot.json'
    predicted=['def solve():\n return 9','def solve():\n return 3','invalid syntax !']
    refs=['def solve():\n return 9','def solve():\n return 4','def solve():\n return 1']
    expected=score_programs(predicted,refs)
    save_generation(output,predicted,refs,{'offset':256,'handwritten_repairs':0})
    scored=score_saved_generation(output)
    assert all(scored[k]==v for k,v in expected.items())
    output.with_suffix('.predictions.jsonl').write_text('{}\n')
    with pytest.raises(AssertionError):score_saved_generation(output)


def test_focal_confirmation_requires_both_seeds_and_subsets():
    from copy import deepcopy
    from homeworks.hw2.focal_confirmation import PLAN, LAUNCH, paired_selection
    def arm(a,b,f=1):
        return {s:{'numeric_correct':v,'prediction_failures':f} for s,v in [('pilot',a),('confirmation',b)]}
    previous={'results':{'control':arm(10,20), 'focal1':arm(11,21)}}
    current={'control':arm(12,22), 'focal1':arm(13,23)}
    assert paired_selection(current,previous)['confirmed']
    current['focal1']=arm(12,24)
    assert not paired_selection(current,previous)['confirmed']
    current['focal1']=arm(13,23,2)
    assert not paired_selection(current,previous)['confirmed']
    current['focal1']=arm(13,23)
    failed_previous=deepcopy(previous)
    failed_previous['results']['focal1']=arm(9,24)
    assert not paired_selection(current,failed_previous)['confirmed']
    assert all(a['seed']==1042 for a in PLAN['arms'].values())
    assert PLAN['final_step']-PLAN['parent_step']==1024
    compile(LAUNCH,'launch.py','exec')


def test_cpu_scoring_process_has_no_torch_and_propagates_failure(tmp_path):
    output=tmp_path/'pilot.json'
    save_generation(output,['def solve():\n return 9'],['def solve():\n return 9'],{'offset':256})
    code="import sys; from pathlib import Path; from homeworks.hw2.numeric_eval import score_saved_generation; score_saved_generation(Path(sys.argv[1])); assert 'torch' not in sys.modules"
    jobs=ScoringJobs();results={'control':{}}
    try:
        jobs.start('control','pilot',[sys.executable,'-c',code,str(output)],output)
        jobs.collect(results,wait=True)
        assert results['control']['pilot']['numeric_correct']==1
        assert len(jobs.completed)==1 and not jobs.pending
        jobs.start('control','fail',[sys.executable,'-c','raise SystemExit(3)'],tmp_path/'fail.json')
        with pytest.raises(subprocess.CalledProcessError):jobs.collect(results,wait=True)
    finally:
        jobs.close()
