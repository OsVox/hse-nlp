import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from homeworks.hw2.grow_ffn import grow_feed_forward


def example():
    torch.manual_seed(17)
    model = T5ForConditionalGeneration(T5Config(vocab_size=31, d_model=32,
        d_kv=8, d_ff=48, num_layers=2, num_decoder_layers=1, num_heads=4,
        dropout_rate=0., feed_forward_proj='gated-gelu', pad_token_id=0,
        eos_token_id=1, decoder_start_token_id=0)).eval()
    batch = dict(input_ids=torch.tensor([[3, 4, 5, 1], [7, 6, 1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        labels=torch.tensor([[9, 10, 1], [8, 11, 1]]))
    return model, batch


def test_preserves_outputs_weights_rng_and_cache_behavior():
    parent, batch = example()
    saved = {k: v.clone() for k, v in parent.state_dict().items()}
    rng = torch.get_rng_state().clone()
    grown = grow_feed_forward(parent, 96, seed=24)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(v, saved[k]) for k, v in parent.state_dict().items())
    assert parent.config.d_ff == 48 and grown.config.d_ff == 96
    assert sum(p.numel() for p in grown.parameters()) - sum(p.numel() for p in parent.parameters()) == 3*3*32*48
    with torch.no_grad():
        torch.testing.assert_close(grown(**batch).logits, parent(**batch).logits, atol=1e-5, rtol=1e-5)
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        expected = parent.generate(**inputs, max_new_tokens=12, do_sample=False)
        for cache in [True, False]:
            assert torch.equal(grown.generate(**inputs, max_new_tokens=12,
                do_sample=False, use_cache=cache), expected)


def test_added_units_can_learn():
    parent, batch = example()
    grown = grow_feed_forward(parent, 96)
    layer = grown.encoder.block[0].layer[-1].DenseReluDense
    optimizer = torch.optim.SGD(grown.parameters(), lr=.01)
    grown(**batch).loss.backward()
    assert layer.wo.weight.grad[:, 48:].abs().sum() > 0
    assert torch.count_nonzero(layer.wi_0.weight.grad[48:]) == 0
    optimizer.step()
    optimizer.zero_grad()
    grown(**batch).loss.backward()
    assert layer.wi_0.weight.grad[48:].abs().sum() > 0
    assert layer.wi_1.weight.grad[48:].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in grown.parameters() if p.grad is not None)


@pytest.mark.parametrize('width', [48, 24])
def test_rejects_non_growth(width):
    parent, _ = example()
    with pytest.raises(ValueError):
        grow_feed_forward(parent, width)
