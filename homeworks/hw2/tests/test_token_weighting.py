import unittest
import torch
import torch.nn.functional as F
from homeworks.hw2.token_weighting import token_weights, weighted_loss


class TokenWeightingTests(unittest.TestCase):
    def test_unit_weights_match_cross_entropy_and_ignore_padding(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        labels = torch.tensor([[1, 2, -100], [3, -100, -100]])
        expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
        actual = weighted_loss(logits, labels, torch.ones(5))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[labels == -100].count_nonzero(), 0)

    def test_weighted_mean(self):
        logits = torch.tensor([[[3., 0.], [3., 0.]]])
        labels = torch.tensor([[0, 1]])
        losses = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction='none')
        torch.testing.assert_close(weighted_loss(logits, labels, torch.tensor([1., 2.])),
                                   (losses[0] + 2*losses[1])/3)

    def test_piece_selection_excludes_identifiers_and_specials(self):
        class Tokenizer:
            all_special_ids = [0]
            pieces = ['1', ' 12', '**', 'variable2', 'return', '1.5', '=']
            def __len__(self): return len(self.pieces)
            def decode(self, ids, **kwargs): return self.pieces[ids[0]]
        self.assertEqual(token_weights(Tokenizer(), 2).tolist(), [1, 2, 2, 1, 1, 2, 1])


if __name__ == '__main__':
    unittest.main()
