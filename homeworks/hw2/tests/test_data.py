"""Prevent held-out questions from returning through another shard."""

from itertools import islice
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa

from homeworks.hw2 import data


def test_question_exclusion_and_global_mixing():
    tables = {
        0: pa.Table.from_pylist([
            {'question': 'reserved', 'code': 'a'},
            {'question': 'reserved2', 'code': 'b'},
            {'question': 'zero', 'code': 'c'},
            {'question': 'one', 'code': 'd'},
        ]),
        1: pa.Table.from_pylist([
            {'question': 'reserved', 'code': 'different reference'},
            {'question': 'two', 'code': 'e'},
            {'question': 'three', 'code': 'f'},
        ]),
    }
    with patch.object(data, 'VALIDATION_ROWS', 2), \
         patch.object(data, 'training_shard', side_effect=lambda i, _: i), \
         patch.object(data.pq, 'read_table', side_effect=lambda i, **_: tables[i]):
        def take(mixed):
            stream = data.training_batches(Path('/unused'), 1, max_shards=2,
                                           mixed_rows_per_shard=mixed)
            return [batch[0] for batch in islice(stream, 4)]
        mixed = take(4)
        assert mixed == take(4)
        assert {r['question'] for r in mixed} == {'zero', 'one', 'two', 'three'}
        assert {r['question'] for r in take(None)} == {'zero', 'one', 'two', 'three'}
