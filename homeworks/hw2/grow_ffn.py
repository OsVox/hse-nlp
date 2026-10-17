"""CPU-only preparation of a wider feed-forward network from our own model.

Existing hidden units are copied. Added input projections are initialized as
in a fresh T5; their output columns start at zero. Thus added units initially
contribute zero, while their output weights can learn on the first update.
This is inspired by function-preserving network growth (Net2Net), not an
implementation of its duplication transform. It has no quality claim.
"""
from copy import deepcopy

import torch
from transformers import T5ForConditionalGeneration


def grow_feed_forward(parent, d_ff, *, seed=42):
    """Return a separate CPU FP32 T5 with a wider gated feed-forward layer.

No parent tensors, global RNG state, optimizer, data cursor, or files change.
Optimizer migration is intentionally outside this transformation; a quality
comparison must control its optimizer initialization in both arms.
"""
    if type(parent) is not T5ForConditionalGeneration:
        raise TypeError('Only the unmodified T5 encoder-decoder is supported')
    if not parent.config.is_gated_act or parent.config.feed_forward_proj != 'gated-gelu':
        raise ValueError('This experiment supports the selected gated-GELU model')
    if not isinstance(d_ff, int) or d_ff <= parent.config.d_ff:
        raise ValueError('New feed-forward width must exceed the original')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in parent.parameters()):
        raise ValueError('Prepare the transformation on CPU in FP32')
    config = deepcopy(parent.config)
    config.d_ff = d_ff
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        grown = T5ForConditionalGeneration(config)
    grown.generation_config = deepcopy(parent.generation_config)
    old_state, new_state = parent.state_dict(), grown.state_dict()
    if old_state.keys() != new_state.keys():
        raise ValueError('Unexpected parameter schema change')
    old_width = parent.config.d_ff
    changed = []
    with torch.no_grad():
        for name, old in old_state.items():
            new = new_state[name]
            if new.shape == old.shape:
                new.copy_(old)
            elif name.endswith(('.DenseReluDense.wi_0.weight', '.DenseReluDense.wi_1.weight')):
                assert old.shape == (old_width, config.d_model)
                assert new.shape == (d_ff, config.d_model)
                new[:old_width].copy_(old)
                changed.append(name)
            elif name.endswith('.DenseReluDense.wo.weight'):
                assert old.shape == (config.d_model, old_width)
                assert new.shape == (config.d_model, d_ff)
                new.zero_()
                new[:, :old_width].copy_(old)
                changed.append(name)
            else:
                raise ValueError('Unexpected resized tensor: ' + name)
    assert len(changed) == 3 * (config.num_layers + config.num_decoder_layers)
    grown.train(parent.training)
    return grown
