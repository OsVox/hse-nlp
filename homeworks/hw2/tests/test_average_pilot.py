import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file, save_file
import torch

from homeworks.hw2.average_pilot import average_checkpoints, passes
from homeworks.hw2.checkpoint_io import sha256


class AverageTests(unittest.TestCase):
    def parent(self, root, name, value):
        p = root/name
        p.mkdir()
        save_file({'weight':torch.tensor([value, value+2])}, str(p/'model.safetensors'))
        for name in ['config.json','generation_config.json','tokenizer.json',
                     'tokenizer_config.json','special_tokens_map.json','step.json']:
            (p/name).write_text('{}')
        (p/'checkpoint.complete.json').write_text(json.dumps({'status':'checkpoint_verified',
            'files':{'model.safetensors':sha256(p/'model.safetensors')}}))
        return p

    def test_exact_mean_and_immutable_parents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parents=[self.parent(root,'a',1.),self.parent(root,'b',3.)]
            hashes=[sha256(p/'model.safetensors') for p in parents]
            average_checkpoints(parents,root/'average')
            torch.testing.assert_close(load_file(str(root/'average/model.safetensors'))['weight'],
                                       torch.tensor([2.,4.]))
            self.assertEqual(hashes,[sha256(p/'model.safetensors') for p in parents])
            with self.assertRaises(AssertionError):
                average_checkpoints(parents,root/'average')

    def test_invalid_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parents=[self.parent(root,'a',1.),self.parent(root,'b',3.)]
            (parents[1]/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaises(AssertionError):
                average_checkpoints(parents,root/'average')
            self.assertFalse((root/'average').exists())

    def test_gate_rejects_tie_and_more_failures(self):
        baseline={'usable_references':250,'numeric_correct':140,'prediction_failures':3}
        self.assertFalse(passes(baseline,baseline))
        self.assertFalse(passes(dict(baseline,numeric_correct=141,prediction_failures=4),baseline))
        self.assertTrue(passes(dict(baseline,numeric_correct=141),baseline))

    def test_explicit_runtime_settings_exception_preserves_parents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parents=[self.parent(root,'a',1.),self.parent(root,'b',3.)]
            (parents[1]/'tokenizer.json').write_text(json.dumps({'padding':{'strategy':'BatchLongest'},
                                                              'truncation':{'max_length':192}}))
            before=[(p/'tokenizer.json').read_bytes() for p in parents]
            with self.assertRaises(AssertionError):
                average_checkpoints(parents,root/'strict')
            average_checkpoints(parents,root/'allowed',allow_runtime_tokenizer_settings=True)
            self.assertEqual(before,[(p/'tokenizer.json').read_bytes() for p in parents])
            self.assertEqual((root/'allowed/tokenizer.json').read_bytes(),before[-1])

    def test_runtime_exception_still_rejects_vocabulary_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parents=[self.parent(root,'a',1.),self.parent(root,'b',3.)]
            (parents[1]/'tokenizer.json').write_text('{"model":{"vocab":{"different":1}}}')
            with self.assertRaises(AssertionError):
                average_checkpoints(parents,root/'bad',allow_runtime_tokenizer_settings=True)

    def test_identical_backend_does_not_allow_changed_special_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parents=[self.parent(root,'a',1.),self.parent(root,'b',3.)]
            (parents[1]/'special_tokens_map.json').write_text('{"eos_token":"changed"}')
            with self.assertRaises(AssertionError):
                average_checkpoints(parents,root/'bad',allow_runtime_tokenizer_settings=True)


if __name__=='__main__':
    unittest.main()
