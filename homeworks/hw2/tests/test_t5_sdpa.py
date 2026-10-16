"""Synthetic numerical tests; no synthetic examples enter model training."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from transformers import T5Config,T5ForConditionalGeneration

from homeworks.hw2.t5_sdpa import enable_sdpa_probe


def model_pair(encoder=2,decoder=2):
    torch.manual_seed(413)
    config=T5Config(vocab_size=43,d_model=64,d_kv=32,d_ff=128,num_heads=2,
        num_layers=encoder,num_decoder_layers=decoder,dropout_rate=0.0,
        feed_forward_proj='gated-gelu',decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    native=T5ForConditionalGeneration(config)
    if decoder>encoder:native.generation_config.cache_implementation='dynamic_full'
    candidate=deepcopy(native)
    assert enable_sdpa_probe(candidate)==encoder+2*decoder
    return native,candidate


def batch():
    return dict(input_ids=torch.tensor([[3,4,5,1,0],[6,7,1,0,0]]),
        attention_mask=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]]),
        labels=torch.tensor([[8,9,10,1],[11,12,1,-100]]))


@pytest.mark.parametrize('layers',[(2,2),(1,3)])
def test_logits_loss_and_all_parameter_gradients(layers):
    native,candidate=model_pair(*layers)
    a=native(**batch(),use_cache=False);b=candidate(**batch(),use_cache=False)
    torch.testing.assert_close(a.logits,b.logits,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(a.loss,b.loss,atol=1e-6,rtol=1e-6)
    a.loss.backward();b.loss.backward()
    native_params=dict(native.named_parameters())
    for name,p in candidate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,native_params[name].grad,atol=2e-6,rtol=3e-4,msg=name)
    for name,p in candidate.named_parameters():
        if 'relative_attention_bias' in name:assert p.grad.abs().sum()>0


@pytest.mark.parametrize('layers',[(2,2),(1,3)])
def test_cached_and_uncached_greedy_ids_and_reload(tmp_path,layers):
    native,candidate=model_pair(*layers);native.eval();candidate.eval()
    data={k:v for k,v in batch().items() if k!='labels'}
    with torch.no_grad():
        expected=native.generate(**data,max_new_tokens=12,do_sample=False,num_beams=1)
        for cache in (True,False):
            actual=candidate.generate(**data,max_new_tokens=12,do_sample=False,num_beams=1,use_cache=cache)
            assert torch.equal(expected,actual)
        candidate.save_pretrained(tmp_path)
        reloaded=T5ForConditionalGeneration.from_pretrained(tmp_path).eval()
        assert set(reloaded.state_dict())==set(native.state_dict())
        enable_sdpa_probe(reloaded)
        assert torch.equal(expected,reloaded.generate(**data,max_new_tokens=12,do_sample=False,num_beams=1))


def test_attention_diagnostics_use_native_fallback():
    native,candidate=model_pair();native.eval();candidate.eval()
    a=native(**batch(),output_attentions=True,use_cache=False)
    with patch('homeworks.hw2.t5_sdpa.F.scaled_dot_product_attention',side_effect=AssertionError('unexpected SDPA')):
        b=candidate(**batch(),output_attentions=True,use_cache=False)
    torch.testing.assert_close(a.logits,b.logits,atol=0,rtol=0)
    for expected,actual in zip(a.decoder_attentions,b.decoder_attentions):
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)


def test_sdpa_dropout_disabled_in_eval_and_preserved_in_training():
    _,candidate=model_pair();original=torch.nn.functional.scaled_dot_product_attention
    for module in candidate.modules():
        if hasattr(module,'n_heads'):module.dropout=.2
    for train,expected in [(False,0.0),(True,.2)]:
        candidate.train(train)
        with patch('homeworks.hw2.t5_sdpa.F.scaled_dot_product_attention',wraps=original) as sdpa:
            candidate(**batch(),use_cache=False)
        assert len(sdpa.call_args_list)==6
        for call in sdpa.call_args_list:
            assert call.kwargs['dropout_p']==expected
            assert call.kwargs['scale']==1.0
