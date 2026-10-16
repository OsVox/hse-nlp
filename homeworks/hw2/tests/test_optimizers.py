import copy
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration
from homeworks.hw2.optimizers import MuonAdamW, make_optimizer, partition_parameters
from homeworks.hw2.train import continuation_scheduler


def tiny():
    torch.manual_seed(7)
    return T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=8,
        d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))


def gradients(model):
    for i, p in enumerate(model.parameters()):
        p.grad = torch.full_like(p, (i+1)/100.)


def test_partition_covers_all_and_excludes_tied_and_relative_embeddings():
    model=tiny(); matrices, other, names=partition_parameters(model)
    assert len(matrices)+len(other)==len(list(model.parameters()))
    assert id(model.shared.weight) in {id(p) for p in other}
    assert id(model.lm_head.weight) in {id(p) for p in other}
    assert all('relative_attention_bias' not in n for n in names['muon'])
    assert any('relative_attention_bias' in n for n in names['adamw'])
    assert all(p.ndim==2 for p in matrices)


def test_wrapper_matches_two_native_optimizers_and_updates_all_parameters():
    a=tiny(); b=copy.deepcopy(a); before=[p.detach().clone() for p in a.parameters()]
    opt=MuonAdamW(a,lr=.001,muon_lr=.002)
    matrices,other,_=partition_parameters(b)
    ref_m=torch.optim.Muon(matrices,lr=.002,weight_decay=.01,
        momentum=.95,nesterov=True,ns_steps=5,adjust_lr_fn='match_rms_adamw')
    ref_a=torch.optim.AdamW(other,lr=.001,weight_decay=.01)
    gradients(a);gradients(b);opt.step();ref_m.step();ref_a.step()
    for old,x,y in zip(before,a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert torch.isfinite(x).all() and not torch.equal(old,x)


def test_resume_keeps_native_state_and_scheduler_group_references():
    a=tiny();opt=MuonAdamW(a,lr=.001,muon_lr=.002)
    sch=continuation_scheduler(opt,learning_rate=.001,start_step=0,end_step=10,warmup=2)
    gradients(a);opt.step();sch.step()
    b=copy.deepcopy(a);other=MuonAdamW(b,lr=.001,muon_lr=.002)
    other_sch=continuation_scheduler(other,learning_rate=.001,start_step=0,end_step=10,warmup=2)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    other_sch.load_state_dict(copy.deepcopy(sch.state_dict()))
    gradients(a);gradients(b);opt.step();other.step();sch.step();other_sch.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert other.param_groups[0] is other.muon.param_groups[0]
    assert other.param_groups[1] is other.adamw.param_groups[0]
    assert other.param_groups[0]['lr']==2*other.param_groups[1]['lr']
    with pytest.raises(ValueError,match='Cannot reinterpret'):
        other.load_state_dict(torch.optim.AdamW(b.parameters()).state_dict())


def test_default_adamw_remains_exact():
    a=tiny();b=copy.deepcopy(a)
    opt=make_optimizer(a,'adamw',learning_rate=.001)
    ref=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.01)
    gradients(a);gradients(b);opt.step();ref.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize('kind', ['adamw', 'muon'])
def test_explicit_reset_retains_step_and_weights_without_loading_old_moments(kind):
    from homeworks.hw2.optimizers import restore_training_state
    model=tiny();old=make_optimizer(model,'adamw',learning_rate=.001)
    old_sch=torch.optim.lr_scheduler.LambdaLR(old,lambda s:1.)
    gradients(model);old.step();old_sch.step()
    before=[p.detach().clone() for p in model.parameters()]
    state=dict(step=73024,optimizer=copy.deepcopy(old.state_dict()),
               scheduler=old_sch.state_dict(),scaler=None)
    opt=make_optimizer(model,kind,learning_rate=.00003,muon_learning_rate=.0001)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    assert restore_training_state(state,opt,sch,scaler,reset=True)==73024
    assert not opt.state
    if kind=='muon':assert not opt.muon.state and not opt.adamw.state
    for x,y in zip(model.parameters(),before):assert torch.equal(x,y)
    sch=continuation_scheduler(opt,learning_rate=.00003,start_step=73024,end_step=74048,warmup=50)
    gradients(model);opt.step();sch.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if kind=='muon':assert opt.param_groups[0]['lr']/opt.param_groups[1]['lr']==pytest.approx(10/3)


def test_unreset_transfer_rejects_adamw_moments():
    from homeworks.hw2.optimizers import restore_training_state
    model=tiny();opt=make_optimizer(model,'muon',learning_rate=.00003,muon_learning_rate=.0001)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
    state=dict(step=73024,optimizer=torch.optim.AdamW(model.parameters()).state_dict())
    with pytest.raises(ValueError,match='Cannot reinterpret'):
        restore_training_state(state,opt,sch,torch.amp.GradScaler('cuda',enabled=False))
