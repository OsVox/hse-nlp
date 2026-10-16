"""Synthetic CPU fixtures only; these records never enter course training."""
import copy
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import PreTrainedTokenizerFast

from homeworks.hw2 import data, full_coverage, train
from homeworks.hw2.checkpoint_io import archive_checkpoint
from homeworks.hw2.full_coverage import FullCoverage, load_index, prepare_index
from homeworks.hw2.input_pipeline import prefetch_batches


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    cache = tmp_path/'cache'
    cache.mkdir()
    monkeypatch.setattr(data, 'VALIDATION_ROWS', 2)
    monkeypatch.setattr(full_coverage, 'VALIDATION_ROWS', 2)
    rows = [
        ['reserved', 'heldout'] + ['q '*i for i in range(1,10)],
        ['reserved', 'x', 'y '*5, 'x', 'z '*9, 'heldout'],
        ['heldout'],
    ]
    for i,questions in enumerate(rows):
        pq.write_table(pa.Table.from_pylist([{'question':q,'code':f'print({j})'}
                       for j,q in enumerate(questions)]),cache/f'part-{i:05d}.parquet')
    index = tmp_path/'index'
    prepare_index(cache,index,3)
    manifest, paths, ids = load_index(cache,index)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    return dict(cache=cache,index=index,manifest=manifest,paths=paths,indices=ids,tokenizer=tokenizer)


def stream(corpus, **kwargs):
    options = dict(seed=42,batch_size=3,window=2,rows_per_shard=2)
    options.update(kwargs)
    return FullCoverage(**{k:corpus[k] for k in ['manifest','paths','indices','tokenizer']}, **options)


def same_batch(left,right):
    assert left['_coverage'] == right['_coverage']
    for key in ['input_ids','labels','attention_mask']:
        assert torch.equal(left[key],right[key])


@pytest.mark.parametrize('window',[1,2,4])
@pytest.mark.parametrize('shuffle',[False,True])
def test_every_row_once_and_resume_at_every_batch(corpus, window, shuffle):
    original = stream(corpus,window=window,shuffle_batch_order=shuffle)
    batches = original.batches()
    seen = []
    epoch = 0
    for _ in range(40):
        # Resume after each consumed batch, including partial windows and epochs.
        restored = stream(corpus,window=window,shuffle_batch_order=shuffle,
                          state=original.checkpoint()['state'])
        expected, actual = next(batches), next(restored.batches())
        same_batch(expected,actual)
        seen.extend(expected['_coverage']['row_ids'])
        original.commit(expected['_coverage'])
        if original.state['cursor'][0] != epoch:
            ids = {(s,int(i)) for s,arr in enumerate(corpus['indices']) for i in arr}
            assert len(seen) == len(set(seen)) == len(ids)
            assert set(seen) == ids
            assert original.checkpoint()['report']['coverage_percent'] == 100
            assert original.state['counts'] == [0,0,0]
            seen = []
            epoch += 1
            if epoch == 2:
                break
    assert epoch == 2
    assert original.checkpoint()['report']['repeated_presentations_since_tracking_started'] == 13


def test_exclusion_preserves_original_duplicate_rows_and_detects_changed_data(corpus):
    assert [x.tolist() for x in corpus['indices']] == [list(range(2,11)),[1,2,3,4],[]]
    path=corpus['paths'][0]
    with path.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_index(corpus['cache'],corpus['index'])


def test_prefetch_does_not_commit_and_repeated_ticket_rejected(corpus):
    sampler=stream(corpus)
    producer=prefetch_batches(sampler.batches(),2)
    batch=next(producer)
    assert sampler.state['presentations']==0
    sampler.commit(batch['_coverage'])
    committed=sampler.checkpoint()['state']
    expected=next(producer)
    assert sampler.checkpoint()['state']==committed
    resumed=stream(corpus,state=committed)
    same_batch(expected,next(resumed.batches()))
    with pytest.raises(ValueError,match='duplicated'):
        sampler.commit(batch['_coverage'])
    producer.close()


def test_resume_rejects_changed_batching_seed_or_bad_counts(corpus):
    sampler=stream(corpus)
    sampler.commit(next(sampler.batches())['_coverage'])
    saved=sampler.checkpoint()['state']
    for options in [dict(seed=43),dict(batch_size=2),dict(window=3),dict(rows_per_shard=3)]:
        with pytest.raises(ValueError,match='changed'):
            stream(corpus,state=saved,**options)
    bad=copy.deepcopy(saved)
    bad['presentations']+=1
    with pytest.raises(ValueError,match='Invalid consumed'):
        stream(corpus,state=bad)


def test_real_training_checkpoint_archive_and_resume(corpus,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    torch.set_num_threads(2)
    common=['train','--cache-dir',str(corpus['cache']),'--coverage-index',str(corpus['index']),
        '--coverage-rows-per-shard','2','--max-shards','3','--batch-size','3',
        '--accumulation','2','--input-pipeline','encoded','--prefetch-batches','2',
        '--group-window','2','--precision','fp32','--save-every','1','--eval-every','100',
        '--warmup','1']
    first=tmp_path/'first'
    monkeypatch.setattr(sys,'argv',common+['--output-dir',str(first),'--steps','2',
        '--d-model','64','--d-ff','128','--encoder-layers','1','--decoder-layers','1','--num-heads','1'])
    train.main()
    saved=json.loads((first/'latest/coverage.json').read_text())
    assert saved.pop('optimizer_step')==2
    expected=stream(corpus)
    expected_batches=expected.batches()
    for _ in range(4):
        expected.commit(next(expected_batches)['_coverage'])
    assert saved==expected.checkpoint()
    assert json.loads((first/'latest/run_config.json').read_text())['coverage_index']==str(corpus['index'])
    parent=tmp_path/'parent'
    manifest=archive_checkpoint(first/'latest',parent)
    assert 'coverage.json' in manifest['files']
    second=tmp_path/'second'
    monkeypatch.setattr(sys,'argv',common+['--output-dir',str(second),'--steps','4',
        '--resume-from',str(parent),'--restart-schedule'])
    train.main()
    for _ in range(4):
        expected.commit(next(expected_batches)['_coverage'])
    actual=json.loads((second/'latest/coverage.json').read_text())
    assert actual.pop('optimizer_step')==4
    assert actual==expected.checkpoint()
    assert actual['report']['completed_full_passes']==1
