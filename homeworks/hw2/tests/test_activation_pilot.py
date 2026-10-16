import copy
import json
import sys
import pytest
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
from homeworks.hw2 import train
from homeworks.hw2.activation_pilot import PLAN,LAUNCH,ACTIVATIONS,quality_selection,verify_records
from homeworks.hw2.tokenizer_pilot import training_command
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_activations_have_identical_parameters_but_distinct_function_and_reload(tmp_path):
    torch.set_num_threads(2)
    tokenizer=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    models=[];inputs=tokenizer(['How much is two plus three?'],return_tensors='pt',return_token_type_ids=False)
    labels=tokenizer(['def solution():\n    return 2 + 3'],return_tensors='pt')['input_ids']
    for activation in ACTIVATIONS.values():
        torch.manual_seed(42)
        m=train.new_model(tokenizer,d_model=64,d_ff=128,encoder_layers=1,
                          decoder_layers=1,num_heads=1,dropout=0,feed_forward_proj=activation).eval()
        loss=m(**inputs,labels=labels).loss;loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        ff=m.encoder.block[0].layer[-1].DenseReluDense
        if activation=='gated-silu':
            probe=torch.linspace(-3,3,17)
            torch.testing.assert_close(ff.act(probe),probe*probe.sigmoid())
        else:
            assert m.config.dense_act_fn=='gelu_new'
        models.append(m)
    assert train.parameter_hash(models[0])==train.parameter_hash(models[1])
    assert not torch.allclose(models[0](**inputs,labels=labels).logits,models[1](**inputs,labels=labels).logits)
    m=models[1];m.save_pretrained(tmp_path);tokenizer.save_pretrained(tmp_path)
    restored=T5ForConditionalGeneration.from_pretrained(tmp_path).eval()
    assert restored.config.feed_forward_proj=='gated-silu'
    assert train.parameter_hash(restored)==train.parameter_hash(m)
    args=dict(max_new_tokens=12,num_beams=1,do_sample=False)
    assert torch.equal(m.generate(**inputs,**args),restored.generate(**inputs,**args))
    assert torch.equal(m.generate(**inputs,use_cache=False,**args),m.generate(**inputs,**args))


def test_parameter_count_and_safe_command():
    class Tokenizer:
        pad_token_id=0;eos_token_id=1
        def __len__(self):return 8192
    for activation in ACTIVATIONS.values():
        with torch.device('meta'):
            m=train.new_model(Tokenizer(),d_model=512,d_ff=1536,encoder_layers=6,decoder_layers=4,num_heads=8,feed_forward_proj=activation)
        assert sum(p.numel() for p in m.parameters())==42481152
    command=training_command(dict(mixed_rows_per_shard=None,coverage_index='/index',steps=2048))
    assert 'None' not in command and '--mixed-rows-per-shard' not in command
    assert '--coverage-index' in command and '--predict-at-checkpoint' in command
    compile(LAUNCH,'launch.py','exec');assert 'timeout=1000' in LAUNCH
    assert PLAN['require_identical_initialization'] and PLAN['steps']==PLAN['save_every']==2048
    assert PLAN['eval_every']==512 and PLAN['mixed_rows_per_shard'] is None


def test_two_subset_gains_and_failure_gate():
    assert quality_selection(dict(control=result(10,20),swiglu=result(11,21)))['shortlist']==['swiglu']
    for candidate in [result(10,22),result(9,30),result(11,21,1)]:
        assert not quality_selection(dict(control=result(10,20),swiglu=candidate))['shortlist']


def test_records_fail_closed_for_different_initial_weights_rows_or_activation():
    records={arm:dict(coverage=dict(optimizer_step=2048,state=dict(row_order_sha256='a'*64),
        report=dict(presentations=262144,unique_rows_since_tracking_started=262144,
        repeated_presentations_since_tracking_started=0,completed_full_passes=0,start_step=0,source_length=128,target_length=192)),
        config=dict(model_config=dict(feed_forward_proj=activation),initial_model_sha256='b'*64)) for arm,activation in ACTIVATIONS.items()}
    assert verify_records(records)['exact_row_order_matched']
    for field in ['weights','rows','activation','count']:
        bad=copy.deepcopy(records)
        if field=='weights':bad['swiglu']['config']['initial_model_sha256']='c'*64
        elif field=='rows':bad['swiglu']['coverage']['state']['row_order_sha256']='d'*64
        elif field=='activation':bad['swiglu']['config']['model_config']['feed_forward_proj']='gated-gelu'
        else:bad['swiglu']['coverage']['report']['presentations']=262145
        with pytest.raises(AssertionError):verify_records(bad)


def test_activation_override_rejected_on_existing_checkpoint(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    tok=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    m=train.new_model(tok,d_model=64,d_ff=128,encoder_layers=1,decoder_layers=1,num_heads=1)
    (tmp_path/'latest').mkdir();m.save_pretrained(tmp_path/'latest');tok.save_pretrained(tmp_path/'latest')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    monkeypatch.setattr(sys,'argv',['train','--output-dir',str(tmp_path),'--feed-forward-proj','gated-silu','--steps','1'])
    with pytest.raises(SystemExit):train.main()
