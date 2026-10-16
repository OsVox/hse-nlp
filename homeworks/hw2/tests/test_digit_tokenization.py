"""Synthetic invariants only: these strings never enter the training data."""
import json

from homeworks.hw2.digit_tokenization import load_training_tokenizer
from homeworks.hw2.input_pipeline import encoded_batches

PATH = 'homeworks/hw2/tokenizer'


def test_digit_boundaries_preserve_ids_text_and_checkpoint_reload(tmp_path):
    old = load_training_tokenizer(PATH)
    new = load_training_tokenizer(PATH, individual_digits=True)
    assert old.get_vocab() == new.get_vocab()
    assert old.all_special_ids == new.all_special_ids
    texts = ['12345.67', '  -100.25 + 3e-12', 'def solve():\n    return 12345 / 67\n',
             'Стоимость: 123 рублей; café 42', 'a12_b34 = 0xFF\r\n', 'plain words only']
    for text in texts:
        ids = new(text)['input_ids']
        assert new.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == text
    assert new.tokenize('12345.67') == list('12345.67')
    assert old(texts[-1]) == new(texts[-1])
    new.save_pretrained(tmp_path)
    restored = load_training_tokenizer(tmp_path)
    repeated = load_training_tokenizer(tmp_path, individual_digits=True)
    assert restored(texts) == repeated(texts) == new(texts)
    assert json.loads(restored.backend_tokenizer.to_str()) == json.loads(repeated.backend_tokenizer.to_str())


def test_prefetch_tokenizer_matches_saved_targets(tmp_path):
    tokenizer = load_training_tokenizer(PATH, individual_digits=True)
    producer = load_training_tokenizer(PATH, individual_digits=True)
    rows = [{'question':'There are 12345 items.', 'code':'def solve():\n    return 12345\n'}]
    batch = next(encoded_batches([rows], producer, 1, 1))
    tokenizer.save_pretrained(tmp_path)
    restored = load_training_tokenizer(tmp_path)
    assert batch['input_ids'][0].tolist() == restored(rows[0]['question'])['input_ids']
    assert batch['labels'][0].tolist() == restored(rows[0]['code'])['input_ids']


def test_screen_requires_both_subsets_and_preserves_budget():
    from homeworks.hw2.digit_screen import PLAN, LAUNCH, select
    def arm(p,c,f=1):
        return {'pilot':{'numeric_correct':p,'prediction_failures':f},
                'confirmation':{'numeric_correct':c,'prediction_failures':f}}
    assert select({'control':arm(10,20),'digits':arm(11,21)})['shortlist']==['digits']
    assert not select({'control':arm(10,20),'digits':arm(10,22)})['shortlist']
    assert not select({'control':arm(10,20),'digits':arm(11,21,2)})['shortlist']
    assert PLAN['final_step']-PLAN['parent_step']==1024
    assert 1024%PLAN['group_window']==0
    assert PLAN['save_every']==PLAN['eval_every']==PLAN['final_step']
    assert PLAN['arms']=={'control':{},'digits':{'individual_digits':True}}
    assert 'timeout=900' in LAUNCH
    compile(LAUNCH,'launch.py','exec')
