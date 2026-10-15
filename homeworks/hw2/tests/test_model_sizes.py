"""Architecture checks without allocating accelerator memory."""
import unittest
from types import SimpleNamespace
import torch
from homeworks.hw2.train import new_model

class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 1
    def __len__(self):
        return 8192

class ModelSizesTest(unittest.TestCase):
    def test_parameter_counts_and_decoder_depth(self):
        for width, enc, expected in [(256,4,9967360),(512,4,35663360),(512,6,42481152)]:
            with self.subTest(width=width, encoder_layers=enc), torch.device('meta'):
                model = new_model(TokenizerStub(), d_model=width, d_ff=width*3,
                                  encoder_layers=enc, num_heads=width//64)
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                self.assertEqual(model.config.num_decoder_layers, 4)
                self.assertEqual(model.config.decoder_start_token_id, 0)
                self.assertEqual(model.config.d_kv, 64)

    def test_invalid_dimensions_rejected(self):
        for args in [dict(d_model=512),dict(encoder_layers=0),dict(dropout=1)]:
            with self.assertRaises(ValueError):
                new_model(TokenizerStub(), **args)

if __name__ == '__main__':
    unittest.main()
