import unittest
import torch
from transformers import T5Config, T5ForConditionalGeneration
from homeworks.hw2.decoder_dropout import decoder_dropout_inputs


class DecoderDropoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = T5ForConditionalGeneration(T5Config(vocab_size=24,d_model=16,
            d_kv=8,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,
            dropout_rate=0,pad_token_id=0,eos_token_id=1,decoder_start_token_id=0))
        self.labels = torch.tensor([[4,5,6,7,8,9,1,-100],[3,10,11,12,13,14,15,1]])

    def test_disabled_and_eval_are_exact_noops(self):
        gen=torch.Generator().manual_seed(42);state=gen.get_state().clone()
        self.assertEqual(decoder_dropout_inputs(self.model,self.labels,0,generator=gen),{})
        self.model.eval()
        self.assertEqual(decoder_dropout_inputs(self.model,self.labels,.5,generator=gen),{})
        self.assertTrue(torch.equal(state,gen.get_state()))

    def test_mask_preserves_labels_specials_and_gradients(self):
        original=self.labels.clone();global_state=torch.random.get_rng_state().clone()
        ids=self.model.prepare_decoder_input_ids_from_labels(self.labels)
        expected=self.model.get_input_embeddings()(ids).detach()
        kwargs=decoder_dropout_inputs(self.model,self.labels,.5,
            generator=torch.Generator().manual_seed(8))
        actual=kwargs['decoder_inputs_embeds']
        self.assertTrue(torch.equal(self.labels,original))
        self.assertTrue(torch.equal(global_state,torch.random.get_rng_state()))
        special=ids.eq(0)|ids.eq(1)
        self.assertTrue(torch.equal(actual.detach()[special],expected[special]))
        changed=actual.detach()[~special]
        zero=changed.eq(0).all(-1)
        self.assertTrue(zero.any() and (~zero).any())
        self.assertTrue(torch.equal(changed[~zero],2*expected[~special][~zero]))
        loss=self.model(input_ids=torch.tensor([[3,1],[4,1]]),labels=self.labels,**kwargs).loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.model.shared.weight.grad).all())
        self.assertGreater(float(self.model.shared.weight.grad.abs().sum()),0)

    def test_invalid_rates_rejected(self):
        for rate in [-.1,1,float('nan')]:
            with self.assertRaises(ValueError):
                decoder_dropout_inputs(self.model,self.labels,rate,generator=torch.Generator())


if __name__ == '__main__':
    unittest.main()
