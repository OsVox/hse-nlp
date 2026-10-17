"""Check a saved full-coverage ledger against its stream cursor without GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(record, config, plan, parent):
    state, report = record['state'], record['report']
    require(state['identity'] == parent['state']['identity'], 'Changed training stream')
    require(config['untruncated_training'] and not config['shuffle_batch_order']
            and config['accumulation'] == 1, 'Unsupported coverage configuration')
    require(report['start_step'] == state['start_step'] == 0, 'Incomplete tracking history')
    counts = report['eligible_by_shard']
    require(counts == parent['report']['eligible_by_shard'], 'Changed eligible rows')
    require(sum(counts) == report['eligible_rows'] == plan['eligible_rows'], 'Wrong dataset size')
    batch_size, chunk = config['batch_size'], config['coverage_rows_per_shard']
    require(batch_size == plan['options']['batch_size'] and
            chunk == plan['options']['coverage_rows_per_shard'], 'Changed batch or round size')
    round_rows = [sum(max(0, min(chunk, n-i*chunk)) for n in counts)
                  for i in range(math.ceil(max(counts)/chunk))]
    round_steps = [math.ceil(n/batch_size) for n in round_rows]
    steps_per_pass = sum(round_steps)
    planned_passes, remainder = divmod(plan['final_step'], steps_per_pass)
    require(planned_passes >= 1 and remainder == 0 and
            plan['final_step'] == config['steps'], 'Wrong full-pass budget')
    epoch, round_id, batch = state['cursor']
    require(0 <= epoch <= planned_passes and 0 <= round_id < len(round_rows) and
            0 <= batch < round_steps[round_id], 'Invalid normalized cursor')
    require(epoch < planned_passes or (round_id == batch == 0), 'Exceeded planned passes')
    expected_step = epoch*sum(round_steps)+sum(round_steps[:round_id])+batch
    expected_rows = epoch*sum(counts)+sum(round_rows[:round_id])+batch*batch_size
    require(record['optimizer_step'] == expected_step, 'Cursor and update count disagree')
    unique_rows = min(expected_rows, sum(counts))
    repeats = expected_rows - unique_rows
    require(expected_rows == state['presentations'] == report['presentations'] and
            unique_rows == report['unique_rows_since_tracking_started'], 'Row count or repetition mismatch')
    require(report['completed_full_passes'] == epoch and
            report['repeated_presentations_since_tracking_started'] == repeats, 'Repeated rows')
    seen = state['counts']
    require(len(seen) == len(counts) and all(0 <= a <= b for a,b in zip(seen,counts)),
            'Invalid per-shard counts')
    visited = counts if epoch else seen
    presented = [epoch*n+a for n,a in zip(counts,seen)]
    require(report['visited_current_epoch_by_shard'] == seen and
            report['unique_visited_by_shard'] == visited and
            report['presentations_by_shard'] == presented and
            sum(visited) == unique_rows and sum(presented) == expected_rows,
            'Per-shard accounting disagrees')
    require(report['source_length'] is None and report['target_length'] is None,
            'Truncated sequences')
    tokens = state['untruncated_tokens']
    require(tokens == report['untruncated_tokens'] and set(tokens) == {'question','code'}
            and all(isinstance(v,int) and v >= expected_rows for v in tokens.values()),
            'Missing full-token accounting')
    return dict(step=expected_step, unique_rows=unique_rows, repeats=repeats,
        presentations=expected_rows, completed_full_passes=epoch, planned_passes=planned_passes,
        cursor=state['cursor'], coverage_percent=100*expected_rows/sum(counts),
        partial_batch_shortfall=expected_step*batch_size-expected_rows,
        complete_pass=epoch >= 1, planned_passes_complete=epoch == planned_passes,
        full_sequences=True, stream_identity=state['identity'],
        scope='Saved-ledger and cursor consistency, not an independent replay of every row ID.')


def read_verified(directory, name):
    manifest=json.loads((directory/'checkpoint.complete.json').read_text())
    require(manifest['status'] == 'checkpoint_verified', 'Incomplete checkpoint')
    content=(directory/name).read_bytes()
    require(hashlib.sha256(content).hexdigest() == manifest['files'][name],
            'Checkpoint file hash mismatch: '+name)
    return json.loads(content)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--ready',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--row-replay',type=Path)
    args=parser.parse_args()
    record=read_verified(args.checkpoint,'coverage.json')
    config=read_verified(args.checkpoint,'run_config.json')
    step=read_verified(args.checkpoint,'step.json')['step']
    require(step == record['optimizer_step'], 'Checkpoint step mismatch')
    ready=json.loads(args.ready.read_text())
    plan=dict(ready['plan'])
    if 'eligible_rows' not in plan:
        plan['eligible_rows']=ready['fresh_preparation']['new_eligible_rows']
    result=verify(record,config,plan,
                  read_verified(args.parent,'coverage.json'))
    if args.row_replay:
        from .replay_coverage_order import compare
        replay=json.loads(args.row_replay.read_text())
        ready=json.loads(args.ready.read_text())
        require(replay['source_commit']==ready['source_commit'], 'Replay source mismatch')
        require(replay['index_manifest_sha256']==ready['fresh_preparation']['index_manifest_sha256'],
                'Replay index mismatch')
        compare(replay['checkpoints'][str(step)],record)
        result.update(independent_row_order_verified=True,
            row_order_sha256=record['state']['row_order_sha256'],
            replay_sha256=hashlib.sha256(args.row_replay.read_bytes()).hexdigest(),
            scope='Saved consumption hash chain, per-shard counts and full-token totals match an independent CPU reconstruction from verified row IDs and cached token lengths.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
