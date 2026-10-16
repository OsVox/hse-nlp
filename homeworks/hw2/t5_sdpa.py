"""Opt-in SDPA probe for our T5 encoder-decoder; production defaults unchanged.

Cache and positional-bias flow follows Hugging Face Transformers v4.57.6
T5Attention.forward (Apache-2.0, Copyright HuggingFace Inc. and Google).
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/t5/modeling_t5.py

Only the attention computation changes. T5 uses unscaled QK scores: scale=1.0
is essential. Learned relative bias must retain its gradient. Backend selection
is left to PyTorch; use sdpa_kernel in a probe to prove a fused backend is used.
"""
from types import MethodType

import torch
from torch.nn import functional as F
import transformers
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5Attention


def sdpa_forward(self, hidden_states, mask=None, key_value_states=None,
                 position_bias=None, past_key_values=None, layer_head_mask=None,
                 query_length=None, use_cache=False, output_attentions=False,
                 cache_position=None):
    # Preserve diagnostics/head masking using the unmodified implementation.
    if output_attentions or layer_head_mask is not None or self.pruned_heads:
        return T5Attention.forward(self, hidden_states, mask=mask,
            key_value_states=key_value_states, position_bias=position_bias,
            past_key_values=past_key_values, layer_head_mask=layer_head_mask,
            query_length=query_length, use_cache=use_cache,
            output_attentions=output_attentions, cache_position=cache_position)
    batch_size, seq_length=hidden_states.shape[:2]
    is_cross=key_value_states is not None
    query=self.q(hidden_states).view(batch_size,-1,self.n_heads,self.key_value_proj_dim).transpose(1,2)
    updated=False
    if isinstance(past_key_values,EncoderDecoderCache):
        updated=past_key_values.is_updated.get(self.layer_idx)
        cache=(past_key_values.cross_attention_cache if is_cross
               else past_key_values.self_attention_cache)
    else:
        cache=past_key_values
    current=key_value_states if is_cross else hidden_states
    if is_cross and past_key_values is not None and updated:
        key=cache.layers[self.layer_idx].keys
        value=cache.layers[self.layer_idx].values
    else:
        key=self.k(current).view(batch_size,-1,self.n_heads,self.key_value_proj_dim).transpose(1,2)
        value=self.v(current).view(batch_size,-1,self.n_heads,self.key_value_proj_dim).transpose(1,2)
        if past_key_values is not None:
            cache_position=cache_position if not is_cross else None
            key,value=cache.update(key,value,self.layer_idx,{'cache_position':cache_position})
            if is_cross and isinstance(past_key_values,EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx]=True
    if position_bias is None:
        key_length=key.shape[-2]
        real_length=query_length if query_length is not None else cache_position[-1]+1
        if self.has_relative_attention_bias:
            position_bias=self.compute_bias(real_length,key_length,
                device=query.device,cache_position=cache_position)[:,:,-seq_length:,:]
        else:
            position_bias=torch.zeros((1,self.n_heads,seq_length,key_length),
                                      device=query.device,dtype=query.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad=True
        if mask is not None:
            position_bias=position_bias+mask[:,:,:,:key.shape[-2]]
    output=F.scaled_dot_product_attention(query,key,value,
        attn_mask=position_bias.to(query.dtype),
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,scale=1.0)
    output=output.transpose(1,2).contiguous().view(batch_size,-1,self.inner_dim)
    return self.o(output),position_bias


def enable_sdpa_probe(model):
    """Patch only this in-memory model. Reloading its state restores native T5."""
    if transformers.__version__!='4.57.6':
        raise ValueError('SDPA probe is audited only against Transformers 4.57.6')
    if getattr(model.config,'model_type',None)!='t5':
        raise ValueError('SDPA probe requires T5')
    count=0
    for module in model.modules():
        if isinstance(module,T5Attention):
            module.forward=MethodType(sdpa_forward,module)
            count+=1
    if not count:
        raise ValueError('No T5 attention modules found')
    return count
