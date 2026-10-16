from pathlib import Path

import pytest
import torch

from homeworks.hw2.train import new_model, nonembedding_hash, tokenizer_for_run


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=1
    def __init__(self,n):self.n=n
    def __len__(self):return self.n


def test_common_initialization_preserves_body_and_tied_embedding_prefix():
    models=[]
    for n in [32,64]:
        torch.manual_seed(42)
        model=new_model(TinyTokenizer(n),d_model=64,d_ff=128,num_heads=1,
                        encoder_layers=1,decoder_layers=1,initialization_vocab_size=64)
        assert model.config.vocab_size==n
        assert model.lm_head.weight.data_ptr()==model.shared.weight.data_ptr()
        assert model(input_ids=torch.tensor([[3,1]]),labels=torch.tensor([[4,1]])).logits.shape[-1]==n
        models.append(model)
    assert nonembedding_hash(models[0])==nonembedding_hash(models[1])
    torch.testing.assert_close(models[0].shared.weight,models[1].shared.weight[:32],rtol=0,atol=0)
    with pytest.raises(ValueError):new_model(TinyTokenizer(64),initialization_vocab_size=32)


def test_tokenizer_override_cannot_replace_trained_checkpoint_tokens(tmp_path):
    custom=tmp_path/'custom';custom.mkdir();(custom/'tokenizer.json').write_text('{}')
    checkpoint=tmp_path/'latest';checkpoint.mkdir()
    assert tokenizer_for_run(checkpoint,custom)==custom
    (checkpoint/'config.json').write_text('{}')
    with pytest.raises(ValueError):tokenizer_for_run(checkpoint,custom)
    (checkpoint/'tokenizer.json').write_text('{}')
    assert tokenizer_for_run(checkpoint)==checkpoint


def test_scratch_pilot_has_bounded_matched_budget_and_strict_gate():
    from homeworks.hw2.tokenizer_pilot import PLAN,LAUNCH,select
    assert PLAN['steps']==PLAN['save_every']==PLAN['eval_every']==2048
    assert PLAN['steps'] % PLAN['group_window']==0
    assert PLAN['steps']*PLAN['batch_size']==PLAN['example_presentations_per_arm']
    assert PLAN['initialization_vocab_size']==16384
    def arm(a,b,f=1):return {s:{'numeric_correct':v,'prediction_failures':f} for s,v in [('pilot',a),('confirmation',b)]}
    result = select({'control':arm(10,20),'code8192':arm(11,21),'code16384':arm(10,22)})
    assert result['shortlist']==['code8192']
    assert not next(row for row in result['ranking'] if row['arm']=='code16384')['eligible']
    assert not select({'control':arm(10,20),'code8192':arm(11,21,2)})['shortlist']
    compile(LAUNCH,'launcher','exec')
