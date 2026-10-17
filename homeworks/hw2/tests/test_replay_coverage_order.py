from copy import deepcopy
import json
from pathlib import Path
import pytest
from homeworks.hw2.replay_coverage_order import compare

ROOT=Path(__file__).resolve().parents[1]/'experiments/selected-full-pass-v2'


def test_independent_replay_matches_saved_consumption():
    replay=json.loads((ROOT/'independent-row-replay.json').read_text())
    for step in (10000,20000,30000,40000):
        record=json.loads((ROOT/f'step-{step:06d}/coverage.json').read_text())
        assert compare(replay['checkpoints'][str(step)],record)


@pytest.mark.parametrize('field,value',[
    ('row_order_sha256','0'*64),
    ('untruncated_tokens',{'question':1,'code':1}),
    ('counts',[0]*60),('cursor',[0,21,56]),
])
def test_replay_rejects_corruption_even_when_total_rows_agree(field,value):
    replay=json.loads((ROOT/'independent-row-replay.json').read_text())
    record=deepcopy(json.loads((ROOT/'step-040000/coverage.json').read_text()))
    record['state'][field]=value
    with pytest.raises(ValueError,match='replay mismatch'):
        compare(replay['checkpoints']['40000'],record)


def test_second_pass_replay_matches_actual_consumption():
    root=ROOT.parent/'selected-scale-v1'
    replay=json.loads((root/'independent-row-replay.json').read_text())
    record=json.loads((root/'segment-01/step-102563/coverage.json').read_text())
    assert compare(replay['checkpoints']['102563'],record)
    assert replay['planned_passes']==2
    assert 184912 not in replay['compared_checkpoint_steps']
    assert replay['checkpoints']['102563']['cursor'][0]==1
    record['state']['row_order_sha256']='0'*64
    with pytest.raises(ValueError,match='row_order_sha256'):
        compare(replay['checkpoints']['102563'],record)
