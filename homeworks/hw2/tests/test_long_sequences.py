"""CPU-only mathematical and sequence-integrity checks; no course training."""
import copy

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from homeworks.hw2.input_pipeline import collate_tokens
from homeworks.hw2.long_sequences import split_batch
from homeworks.hw2.token_weighting import weighted_loss


def fixture_batch():
    return collate_tokens([[2,3,1],[4,5,6,7,1],[9,1]],
                          [[3,4,1],[2,8,9,10,11,1],[7,8,9,1]],pad_id=0,eos_id=1)


@pytest.mark.parametrize('mode',['native','weighted','balanced','smooth','focal'])
def test_split_gradients_equal_logical_batch(mode):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=T5ForConditionalGeneration(T5Config(vocab_size=16,d_model=16,d_kv=8,
        d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0,
        pad_token_id=0,eos_token_id=1,decoder_start_token_id=0))
    split=copy.deepcopy(model)
    batch=fixture_batch()
    weights=None if mode=='native' else torch.arange(1,17).float()
    options=dict(example_balanced=mode=='balanced',smoothing=.1 if mode=='smooth' else 0,
                 focal_gamma=1.5 if mode=='focal' else 0)
    def loss_for(m,b):
        output=m(**b,use_cache=False)
        return output.loss if weights is None else weighted_loss(output.logits,b['labels'],weights,**options)
    reference=loss_for(model,batch)
    reference.backward()
    total=0.
    chunks=list(split_batch(batch,token_budget=11,attention_budget=100,
        weights=weights,example_balanced=mode=='balanced'))
    assert len(chunks)==3 and sum(f for _,f in chunks)==pytest.approx(1.)
    for part,fraction in chunks:
        value=loss_for(split,part)*fraction
        total+=value.item()
        value.backward()
    assert total==pytest.approx(reference.item(),rel=2e-6)
    for a,b in zip(model.parameters(),split.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=1e-6)


def test_long_outlier_preserved_and_budget_failure_is_explicit():
    batch=collate_tokens([[2]*740+[1],[3,1]],[[4]*1207+[1],[5,1]],pad_id=0,eos_id=1)
    chunks=list(split_batch(batch,token_budget=2000,attention_budget=3000000))
    assert len(chunks)==2
    assert chunks[0][0]['input_ids'].shape==(1,741)
    assert chunks[0][0]['labels'].shape==(1,1208)
    assert torch.equal(chunks[0][0]['labels'][0],batch['labels'][0])
    # A later over-budget row must be detected before any chunk is yielded.
    reversed_batch={k:v.flip(0) for k,v in batch.items()}
    with pytest.raises(ValueError,match='Single example exceeds'):
        next(split_batch(reversed_batch,token_budget=100,attention_budget=10000))
