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


def grow_adamw_optimizer(parent, grown, optimizer):
    """Copy mature AdamW history, padding only new coordinates with zeros.

The per-tensor step remains the parent's step, including for new coordinates.
Thus this does not give those coordinates a fresh Adam bias-correction clock.
It preserves the effective history of old coordinates; quality testing must
evaluate the learning rate for the added coordinates. No schedule, random
state or data cursor is changed by this function.
"""
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError('Only native AdamW state can be migrated')
    old_params, new_params = dict(parent.named_parameters()), dict(grown.named_parameters())
    if old_params.keys() != new_params.keys():
        raise ValueError('Parameter names changed')
    if grown.config.d_ff <= parent.config.d_ff:
        raise ValueError('Expected a wider feed-forward model')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32
           for p in list(old_params.values()) + list(new_params.values())):
        raise ValueError('Migrate optimizer state on CPU in FP32')
    by_id = {id(p): name for name, p in old_params.items()}
    visited, groups = [], []
    for group in optimizer.param_groups:
        if group.get('fused') or group.get('capturable') or group.get('differentiable'):
            raise ValueError('Unsupported AdamW execution mode for CPU preparation')
        names = [by_id.get(id(p)) for p in group['params']]
        if None in names:
            raise ValueError('Optimizer contains parameters from another model')
        visited.extend(names)
        groups.append({**deepcopy({k: v for k, v in group.items() if k != 'params'}),
                       'params': [new_params[n] for n in names]})
    if len(set(visited)) != len(visited) or set(visited) != set(old_params):
        raise ValueError('Optimizer must cover every model parameter exactly once')
    result = torch.optim.AdamW(groups)
    for name, old in old_params.items():
        state = optimizer.state.get(old, {})
        expected = {'step', 'exp_avg', 'exp_avg_sq'}
        if not expected.issubset(state) or set(state) - (expected | {'max_exp_avg_sq'}):
            raise ValueError('Missing or unsupported AdamW state for ' + name)
        new = new_params[name]
        migrated = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                raise ValueError('Expected CPU tensor optimizer state')
            if key == 'step':
                if value.ndim != 0 or not torch.isfinite(value) or value.item() < 1:
                    raise ValueError('Expected a trained optimizer step')
                migrated[key] = value.clone()
                continue
            if value.shape != old.shape or value.dtype != old.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid moment tensor for ' + name)
            if old.shape == new.shape:
                migrated[key] = value.clone()
                continue
            padded = torch.zeros_like(new)
            if name.endswith(('.DenseReluDense.wi_0.weight', '.DenseReluDense.wi_1.weight')):
                if new.shape != (grown.config.d_ff, parent.config.d_model):
                    raise ValueError('Unexpected grown input projection')
                padded[:old.shape[0]].copy_(value)
            elif name.endswith('.DenseReluDense.wo.weight'):
                if new.shape != (parent.config.d_model, grown.config.d_ff):
                    raise ValueError('Unexpected grown output projection')
                padded[:, :old.shape[1]].copy_(value)
            else:
                raise ValueError('Unexpected resized optimizer tensor: ' + name)
            migrated[key] = padded
        result.state[new] = migrated
    return result
