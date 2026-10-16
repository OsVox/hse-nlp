"""Learned source-copy mixture on our own scratch-trained T5 Transformer.

The reference-token likelihood is the sole fitting objective. This module does
not parse, repair or execute generated programs, or use external model weights.
"""
import torch
from torch import nn
from transformers import T5ForConditionalGeneration


def normalized_heads(attentions, source_ids, attention_mask, pad_id, eos_id):
    # B,H,T,S -> B,T,H,S; exclude padding and EOS from the copying vocabulary.
    heads=torch.cat(attentions,dim=1).float().permute(0,2,1,3)
    valid=attention_mask.bool() & source_ids.ne(pad_id) & source_ids.ne(eos_id)
    heads=heads*valid[:,None,None,:]
    denominator=heads.sum(-1,keepdim=True)
    return heads/denominator.clamp_min(1e-12),valid.any(-1)


class CopyT5(T5ForConditionalGeneration):
    def __init__(self,config):
        super().__init__(config)
        self.copy_gate=nn.Linear(config.d_model,1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias,-4.)
        self.copy_head_logits=nn.Parameter(torch.zeros(config.num_decoder_layers*config.num_heads))
        self.config.copy_adapter=True
        self.copy_enabled=True

    def copy_probability(self,hidden):
        return torch.sigmoid(self.copy_gate(hidden.float()))

    def forward(self,input_ids=None,attention_mask=None,decoder_input_ids=None,
                encoder_outputs=None,labels=None,copy_source_ids=None,
                output_attentions=None,output_hidden_states=None,return_dict=None,**kwargs):
        if not self.copy_enabled:
            return super().forward(input_ids=input_ids,attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,encoder_outputs=encoder_outputs,labels=labels,
                output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                return_dict=return_dict,**kwargs)
        source_ids=copy_source_ids if copy_source_ids is not None else input_ids
        if source_ids is None:raise ValueError('Copy decoding requires original source token IDs')
        if attention_mask is None:attention_mask=source_ids.ne(self.config.pad_token_id).long()
        out=super().forward(input_ids=input_ids,attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,encoder_outputs=encoder_outputs,labels=labels,
            output_attentions=True,output_hidden_states=True,return_dict=True,**kwargs)
        heads,has_source=normalized_heads(out.cross_attentions,source_ids,attention_mask,
            self.config.pad_token_id,self.config.eos_token_id)
        pointer=(heads*self.copy_head_logits.softmax(0)[None,None,:,None]).sum(2)
        distribution=torch.zeros_like(out.logits,dtype=torch.float32)
        distribution.scatter_add_(2,source_ids[:,None,:].expand_as(pointer),pointer)
        gate=self.copy_probability(out.decoder_hidden_states[-1])*has_source[:,None,None]
        mixed=(1-gate)*out.logits.float().softmax(-1)+gate*distribution
        out.logits=mixed.clamp_min(torch.finfo(torch.float32).tiny).log()
        if labels is not None:
            out.loss=nn.functional.cross_entropy(out.logits.transpose(1,2),labels,ignore_index=-100)
        if not output_attentions:
            out.decoder_attentions=None;out.cross_attentions=None
        if not output_hidden_states:out.decoder_hidden_states=None;out.encoder_hidden_states=None
        return out if return_dict is not False else out.to_tuple()

    def generate(self,inputs=None,**kwargs):
        source=kwargs.get('input_ids',inputs)
        if source is None:raise ValueError('Pass source input_ids to copy generation')
        return super().generate(inputs,copy_source_ids=source,**kwargs)


def from_base_checkpoint(path):
    """Add the small copy module without changing any existing base parameter."""
    base=T5ForConditionalGeneration.from_pretrained(path,local_files_only=True,attn_implementation='eager')
    model=CopyT5(base.config)
    missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
    assert set(missing)=={'copy_gate.weight','copy_gate.bias','copy_head_logits'} and not unexpected
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('copy_'))
    model.eval()
    return model
