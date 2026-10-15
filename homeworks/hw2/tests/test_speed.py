"""Проверки сохранения данных и результатов после оптимизации."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast

from homeworks.hw2.speed import group_batches, precision_dtype
from homeworks.hw2.train import encode_batch, new_model
from homeworks.hw2.inference import generate_predictions


class SpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')

    def test_grouping_preserves_rows_and_reduces_padding(self):
        rows = [{'question': 'number ' * n, 'code': 'x = 1\n' * n}
                for n in [1, 30, 2, 40, 3, 50, 4, 60, 5]]
        batches = [rows[i:i+2] for i in range(0, len(rows), 2)]
        grouped = list(group_batches(iter(batches), self.tokenizer, 2, 5))
        self.assertEqual(Counter(tuple(r.items()) for b in grouped for r in b),
                         Counter(tuple(r.items()) for r in rows))
        self.assertEqual(list(group_batches(iter(batches), self.tokenizer, 2, 1)), batches)
        def slots(bs):
            return sum(encode_batch(b, self.tokenizer, torch.device('cpu'))['labels'].numel()
                       for b in bs)
        self.assertLess(slots(grouped), slots(batches))

    def test_precision_selection(self):
        self.assertEqual(precision_dtype('auto', torch.device('cpu')), torch.float32)
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(precision_dtype('auto', torch.device('cuda')), torch.bfloat16)
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            self.assertEqual(precision_dtype('auto', torch.device('cuda')), torch.float16)
            with self.assertRaises(ValueError):
                precision_dtype('bf16', torch.device('cuda'))

    def test_no_cache_training_and_cached_generation(self):
        torch.manual_seed(42)
        model = new_model(self.tokenizer)
        batch = encode_batch([{'question': 'add two numbers', 'code': 'def solve():\n    return 2'}],
                             self.tokenizer, torch.device('cpu'))
        model.eval()
        cached = model(**batch, use_cache=True)
        uncached = model(**batch, use_cache=False)
        torch.testing.assert_close(cached.logits, uncached.logits)
        self.assertIsNone(uncached.past_key_values)
        uncached.loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        self.assertTrue(torch.equal(model.generate(**inputs, max_new_tokens=8, use_cache=True),
                                    model.generate(**inputs, max_new_tokens=8, use_cache=False)))

    def test_inference_preserves_order_and_duplicates(self):
        questions = ['long question ' * 10, 'short', 'short', 'medium question']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pq.write_table(pa.table({'question': questions}), root / 'test.parquet')
            # Подмена только генерации: проверяем токенизацию, сортировку и запись.
            def generate(model, input_ids, **kwargs):
                return input_ids
            with patch('transformers.T5ForConditionalGeneration.generate', generate):
                count, _ = generate_predictions(Path('homeworks/hw2/model'), root/'out.csv', root,
                                                batch_size=2, max_new_tokens=8)
            with (root/'out.csv').open(newline='') as f:
                actual = [r['code'] for r in csv.DictReader(f)]
            expected = [s.strip() for s in self.tokenizer.batch_decode(
                self.tokenizer(questions, truncation=True, max_length=128)['input_ids'],
                skip_special_tokens=True)]
            self.assertEqual(count, 4)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
