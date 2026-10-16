import random
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from homeworks.hw2.question_frequency import FrequencyIndex, build, keep_row


def fixture_data(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    for shard, questions in enumerate([['heldout','repeat','single'],['heldout','repeat','repeat','other']]):
        pq.write_table(pa.table({'question':questions,'code':['print(1)']*len(questions)}),
                       cache/f'part-{shard:05d}.parquet')
    return cache


def test_cross_shard_counts_exclude_all_heldout_copies(tmp_path):
    cache=fixture_data(tmp_path);out=tmp_path/'index'
    result=build(cache,out,shards=2,heldout_rows=1,sample_per_shard=10)
    index=FrequencyIndex(out,cache,shards=2,heldout_rows=1)
    assert index.arrays[0].tolist()==[0,3,1]
    assert index.arrays[1].tolist()==[0,3,3,1]
    assert result['eligible_rows']==5 and result['unique_questions']==3
    assert result['arms']['1']['expected_acceptance']==pytest.approx(3/5)
    assert result['arms']['1']['expected_repeated_question_fraction']==pytest.approx(1/3)


def test_modified_source_or_index_is_rejected(tmp_path):
    cache=fixture_data(tmp_path);out=tmp_path/'index'
    build(cache,out,shards=2,heldout_rows=1)
    np.save(out/'part-00000.npy',np.array([0,1,1],dtype=np.uint32))
    with pytest.raises(ValueError,match='checksum'):
        FrequencyIndex(out,cache,shards=2,heldout_rows=1)


def test_probabilities_and_deterministic_replay():
    def take():
        rng=random.Random(7)
        return [keep_row(4,1,rng) for _ in range(10000)]
    a=take();assert a==take()
    assert 2300 < sum(a) < 2700
    assert keep_row(1,1,random.Random(0))
    assert not keep_row(0,0,random.Random(0))
    assert keep_row(100,0,random.Random(0))


def test_training_stream_preserves_targets_and_zero_power_baseline(tmp_path):
    from itertools import islice
    from unittest.mock import patch
    from homeworks.hw2 import data
    cache=fixture_data(tmp_path);out=tmp_path/'index'
    build(cache,out,shards=2,heldout_rows=1)
    index=FrequencyIndex(out,cache,shards=2,heldout_rows=1)
    with patch.object(data,'VALIDATION_ROWS',1):
        def take(power,idx):
            return list(islice(data.training_batches(cache,2,max_shards=2,
                mixed_rows_per_shard=10,frequency_index=idx,frequency_power=power),10))
        assert take(0,None)==take(0,index)
        rows=take(1,index)
        assert rows==take(1,index)
        assert all(r['question']!='heldout' and r['code']=='print(1)' for b in rows for r in b)
