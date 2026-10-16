import tempfile
import unittest
import torch
from transformers import T5Config,T5ForConditionalGeneration
from homeworks.hw2.copy_adapter import CopyT5,normalized_heads


class CopyAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=CopyT5(T5Config(vocab_size=24,d_model=16,d_kv=8,d_ff=32,
            num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0,
            pad_token_id=0,eos_token_id=1,decoder_start_token_id=0)).eval()
        self.ids=torch.tensor([[3,3,4,1],[7,8,1,0]])
        self.mask=self.ids.ne(0).long()

    def test_normalization_duplicate_copy_and_empty_source(self):
        a=torch.ones(2,2,3,4)/4
        h,valid=normalized_heads((a,),self.ids,self.mask,0,1)
        self.assertTrue(torch.allclose(h.sum(-1),torch.ones(2,3,2)))
        self.assertTrue(valid.all())
        out=self.model(input_ids=self.ids,attention_mask=self.mask,labels=torch.tensor([[3,4,1],[8,7,1]]))
        self.assertTrue(torch.allclose(out.logits.exp().sum(-1),torch.ones(2,3),atol=1e-6))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(torch.isfinite(self.model.copy_gate.weight.grad).all())
        empty=self.model(input_ids=torch.tensor([[1,0]]),labels=torch.tensor([[3,1]]))
        self.assertTrue(torch.allclose(empty.logits.exp().sum(-1),torch.ones(1,2),atol=1e-6))

    def test_cached_greedy_and_save_reload(self):
        options=dict(input_ids=self.ids,attention_mask=self.mask,max_new_tokens=8,do_sample=False,num_beams=1)
        expected=self.model.generate(**options,use_cache=False)
        self.assertTrue(torch.equal(expected,self.model.generate(**options,use_cache=True)))
        with tempfile.TemporaryDirectory() as p:
            self.model.save_pretrained(p)
            other=CopyT5.from_pretrained(p,local_files_only=True).eval()
            self.assertTrue(torch.equal(expected,other.generate(**options,use_cache=True)))

    def test_fitting_likelihood_matches_full_distribution(self):
        labels=torch.tensor([[3,4,1],[8,7,1]])
        base=T5ForConditionalGeneration.forward(self.model,input_ids=self.ids,attention_mask=self.mask,
            labels=labels,output_attentions=True,output_hidden_states=True,return_dict=True)
        heads,_=normalized_heads(base.cross_attentions,self.ids,self.mask,0,1)
        match=labels[:,:,None].eq(self.ids[:,None,:])
        mass=(heads*match[:,:,None,:]).sum(-1)
        copy=(mass*self.model.copy_head_logits.softmax(0)).sum(-1)
        gen=base.logits.softmax(-1).gather(-1,labels[:,:,None]).squeeze(-1)
        gate=self.model.copy_probability(base.decoder_hidden_states[-1]).squeeze(-1)
        expected=(1-gate)*gen+gate*copy
        full=self.model(input_ids=self.ids,attention_mask=self.mask,labels=labels)
        actual=full.logits.exp().gather(-1,labels[:,:,None]).squeeze(-1)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))
        self.model.copy_enabled=False
        unchanged=self.model(input_ids=self.ids,attention_mask=self.mask,labels=labels)
        self.assertTrue(torch.equal(base.logits,unchanged.logits))


if __name__=='__main__':unittest.main()
