import pytest
from homeworks.hw2.depth_pilot import PLAN, select
from homeworks.hw2.tokenizer_pilot import verify_initializations
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_depth_selection_requires_both_gains_and_no_extra_failures():
    r=select(dict(control=result(10,20),encoder_heavy=result(11,21),decoder_heavy=result(9,30)))
    assert r['shortlist']==['encoder_heavy']
    r=select(dict(control=result(10,20),encoder_heavy=result(11,21,1),decoder_heavy=result(10,20)))
    assert not r['shortlist']


def test_initialization_mismatch_still_rejected_by_default():
    verify_initializations({'a':'x','b':'x'}, {})
    with pytest.raises(AssertionError):verify_initializations({'a':'x','b':'y'}, {})
    verify_initializations({'a':'x','b':'y'}, PLAN)
    with pytest.raises(AssertionError):verify_initializations({'a':''}, PLAN)
    assert all(o['optimizer']=='muon' for o in PLAN['arm_options'].values())
    counts=list(PLAN['expected_parameters'].values())
    assert max(abs(n/counts[0]-1) for n in counts)<.0032


def test_deeper_decoder_cached_generation_matches_uncached_after_reload(tmp_path):
    import torch
    from transformers import T5ForConditionalGeneration
    from homeworks.hw2.train import new_model
    class TinyTokenizer:
        pad_token_id=0
        eos_token_id=1
        def __len__(self):return 32
    torch.manual_seed(42)
    model=new_model(TinyTokenizer(),d_model=64,d_ff=128,encoder_layers=1,
                    decoder_layers=2,num_heads=1,dropout=0).eval()
    inputs=dict(input_ids=torch.tensor([[2,3,4,1]]),max_new_tokens=6,
                do_sample=False,num_beams=1)
    with torch.no_grad():
        cached=model.generate(**inputs,use_cache=True)
        uncached=model.generate(**inputs,use_cache=False,cache_implementation=None)
    assert torch.equal(cached,uncached)
    model.save_pretrained(tmp_path)
    restored=T5ForConditionalGeneration.from_pretrained(tmp_path).eval()
    assert restored.generation_config.cache_implementation=='dynamic_full'
    with torch.no_grad():assert torch.equal(cached,restored.generate(**inputs,use_cache=True))
