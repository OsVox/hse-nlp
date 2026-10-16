"""Exact batch equivalence and background-worker failure handling.

All text below is synthetic test-only input; it is never used for training.
"""
from threading import enumerate as threads
import time

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from homeworks.hw2.input_pipeline import encoded_batches, prefetch_batches, to_device
from homeworks.hw2.speed import group_batches
from homeworks.hw2.train import encode_batch


@pytest.mark.parametrize('window', [1, 2, 32])
@pytest.mark.parametrize('depth', [0, 2])
def test_matches_legacy_tensors(window, depth):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    rows = [{'question': 'number ' * n, 'code': 'x = 1\n' * n}
            for n in [1, 30, 2, 150, 3, 50, 4, 300, 5]]
    rows += [{'question': '<pad> text </s>', 'code': '<pad><pad>value</s>'},
             {'question': '', 'code': ''}, {'question': 'number ', 'code': 'x = 1\n'}]
    # Deliberately irregular batches, a final partial window, and duplicate rows.
    raw = [rows[:3], rows[3:5], rows[5:9], rows[9:11], rows[11:]]
    expected = [encode_batch(b, tokenizer, torch.device('cpu'))
                for b in group_batches(iter(raw), tokenizer, 3, window)]
    actual = list(prefetch_batches(encoded_batches(iter(raw), tokenizer, 3, window), depth))
    assert len(expected) == len(actual)
    for old, new in zip(expected, actual):
        assert old.keys() == new.keys()
        for key in old:
            assert torch.equal(old[key], new[key]), key
        assert all(torch.equal(new[k], v) for k, v in to_device(new, torch.device('cpu')).items())


def test_prefetch_propagates_failure_and_preserves_prior_batches():
    def stream():
        yield {'input_ids': torch.tensor([7])}
        raise ValueError('incomplete input shard')
    prepared = prefetch_batches(stream(), 2)
    assert next(prepared)['input_ids'].item() == 7
    with pytest.raises(ValueError, match='incomplete input shard'):
        next(prepared)


def test_closing_consumer_releases_full_queue_worker():
    def stream():
        while True:
            yield {'input_ids': torch.tensor([1])}
    prepared = prefetch_batches(stream(), 1)
    next(prepared)
    prepared.close()
    deadline = time.monotonic() + 2
    while any(t.name == 'hw2-input-prefetch' for t in threads()) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not any(t.name == 'hw2-input-prefetch' for t in threads())


def test_shuffled_batch_order_preserves_exact_groups_and_is_reproducible():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    rows = [{'question': 'number ' * n, 'code': f'def solve():\n    return {n}\n'}
            for n in range(1, 21)]
    raw = [rows[i:i+2] for i in range(0, len(rows), 2)]

    def batches(seed):
        return list(encoded_batches(raw, tokenizer, 2, 5, shuffle_batch_seed=seed))

    def identity(batch):
        return tuple((k, tuple(v.shape), tuple(v.flatten().tolist())) for k, v in sorted(batch.items()))

    original = [identity(b) for b in batches(None)]
    shuffled = [identity(b) for b in batches(42)]
    assert shuffled == [identity(b) for b in batches(42)]
    assert shuffled != original
    for start in (0, 5):
        assert sorted(shuffled[start:start+5]) == sorted(original[start:start+5])
