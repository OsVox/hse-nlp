"""CPU-prepared full-pass sampler, resumable after the last consumed batch.

Rows retain their original (shard, row) identity. Per-shard permutations are
partitioned into disjoint chunks, mixed across shards, then grouped by length.
Prefetch advances an iterator, never the separately committed coverage ledger.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .checkpoint_io import sha256
from .data import SHARDS, VALIDATION_ROWS, training_shard, validation_examples
from .input_pipeline import collate_tokens


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_index(cache, destination, shards=SHARDS):
    """Read all questions and exclude every held-out copy, before GPU allocation."""
    if not 1 <= shards <= SHARDS:
        raise ValueError('Invalid shard count')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    heldout = {r['question'] for r in validation_examples(cache)}
    records = []
    for shard in range(shards):
        path = training_shard(shard, cache)
        questions = pq.read_table(path, columns=['question'])['question'].to_pylist()
        eligible = np.asarray([i for i,q in enumerate(questions)
            if (shard != 0 or i >= VALIDATION_ROWS) and q not in heldout], dtype=np.int64)
        index_path = destination/f'{shard:05d}.npy'
        np.save(index_path, eligible, allow_pickle=False)
        records.append(dict(shard=shard, filename=path.name, rows=len(questions),
            eligible=len(eligible), excluded=len(questions)-len(eligible),
            parquet_sha256=sha256(path), index_sha256=sha256(index_path)))
    manifest = dict(version=1, shards=records, heldout_question_sha256=digest(sorted(heldout)))
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_index(cache, directory):
    directory, cache = Path(directory), Path(cache)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['version'] != 1:
        raise ValueError('Unsupported index version')
    indices, paths = [], []
    for i,record in enumerate(manifest['shards']):
        if record['shard'] != i or Path(record['filename']).name != record['filename']:
            raise ValueError('Invalid shard identity')
        path, index = cache/record['filename'], directory/f'{i:05d}.npy'
        if sha256(path) != record['parquet_sha256'] or sha256(index) != record['index_sha256']:
            raise ValueError('Coverage index/data checksum mismatch')
        ids = np.load(index, allow_pickle=False)
        if (ids.dtype != np.int64 or ids.ndim != 1 or len(ids) != record['eligible'] or
                (len(ids) and (ids[0] < 0 or ids[-1] >= record['rows'])) or
                np.any(ids[1:] <= ids[:-1])):
            raise ValueError('Invalid eligible row index')
        indices.append(ids)
        paths.append(path)
    if not indices or not sum(map(len, indices)):
        raise ValueError('No eligible training rows')
    return manifest, paths, indices


class FullCoverage:
    def __init__(self, manifest, paths, indices, tokenizer, *, seed, batch_size,
                 window=32, rows_per_shard=4096, shuffle_batch_order=False,
                 state=None, start_step=0, target_length=192):
        if min(batch_size, window, rows_per_shard) < 1 or seed < 0:
            raise ValueError('Invalid coverage stream configuration')
        self.paths, self.indices = paths, indices
        # Fast tokenizers mutate truncation settings; own one per producer.
        self.tokenizer = copy.deepcopy(tokenizer)
        self.counts = [len(x) for x in indices]
        self.excluded = [r['excluded'] for r in manifest['shards']]
        if not sum(self.counts):
            raise ValueError('No eligible rows')
        self.seed, self.batch_size, self.window = seed, batch_size, window
        self.chunk = rows_per_shard
        if target_length < 192:
            raise ValueError('Target length must be at least the fixed grouping cap of 192')
        self.target_length = target_length
        self.shuffle_batches = shuffle_batch_order
        self.rounds = math.ceil(max(self.counts)/self.chunk)
        token_config = json.loads(tokenizer.backend_tokenizer.to_str())
        token_config.pop('padding', None)
        token_config.pop('truncation', None)
        identity = dict(version=1, manifest=manifest, seed=seed,
            batch_size=batch_size, window=window, rows_per_shard=rows_per_shard,
            shuffle_batch_order=shuffle_batch_order, tokenizer=token_config,
            padding_side=tokenizer.padding_side, numpy_version=np.__version__)
        # Preserve existing default-stream fingerprints; changed limits cannot
        # resume an old ledger silently. Grouping always uses the old 192 cap.
        if target_length != 192:
            identity.update(target_length=target_length, grouping_target_length=192)
        self.identity = digest(identity)
        if tokenizer.padding_side != 'right':
            raise ValueError('Coverage pipeline requires right padding')
        self.state = copy.deepcopy(state) if state else dict(
            identity=self.identity, cursor=[0, 0, 0], counts=[0]*len(indices),
            presentations=0, row_order_sha256='0'*64, start_step=start_step)
        if self.state['identity'] != self.identity:
            raise ValueError('Cannot resume a changed data/tokenization/batching stream')
        epoch, round_id, batch = self.state['cursor']
        if (epoch < 0 or not 0 <= round_id < self.rounds or
                not 0 <= batch < self.batches_in_round(round_id) or
                len(self.state['counts']) != len(self.counts) or
                any(not 0 <= a <= b for a,b in zip(self.state['counts'], self.counts)) or
                self.state['presentations'] != epoch*sum(self.counts)+sum(self.state['counts'])):
            raise ValueError('Invalid consumed-stream checkpoint state')
        window_id, within_window = divmod(batch,self.window)
        width = self.batch_size*self.window
        remaining = min(width,self.round_size(round_id)-window_id*width)
        group_sizes = [min(self.batch_size,remaining-i)
                       for i in range(0,remaining,self.batch_size)]
        if self.shuffle_batches:
            self.rng(2,epoch,round_id,window_id).shuffle(group_sizes)
        consumed_prefix = (sum(self.round_size(i) for i in range(round_id))+
                           window_id*width+sum(group_sizes[:within_window]))
        if sum(self.state['counts']) != consumed_prefix:
            raise ValueError('Coverage counts disagree with the saved cursor')
        if any(not min(round_id*self.chunk,n) <= seen <= min((round_id+1)*self.chunk,n)
               for seen,n in zip(self.state['counts'],self.counts)):
            raise ValueError('Per-shard counts disagree with the saved round')

    def rng(self, *parts):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, *parts])))

    def round_size(self, round_id):
        return sum(min(self.chunk, max(0, n-round_id*self.chunk)) for n in self.counts)

    def batches_in_round(self, round_id):
        return math.ceil(self.round_size(round_id)/self.batch_size)

    def epoch_permutations(self, epoch):
        return [self.rng(0,epoch,i).permutation(ids) for i,ids in enumerate(self.indices)]

    def round_plan(self, epoch, round_id, permutations):
        """Original row IDs in the exact pre-grouping order; also usable for CPU audits."""
        parts = []
        for shard,ids in enumerate(permutations):
            selected = ids[round_id*self.chunk:(round_id+1)*self.chunk]
            parts.append(np.column_stack((np.full(len(selected),shard,dtype=np.int64),selected)))
        plan = np.concatenate(parts)
        return plan[self.rng(1,epoch,round_id).permutation(len(plan))]

    def batches(self):
        cursor = list(self.state['cursor'])
        cached_epoch, permutations = None, None
        while True:
            epoch, round_id, batch_offset = cursor
            if cached_epoch != epoch:
                permutations = self.epoch_permutations(epoch)
                cached_epoch = epoch
            plan = self.round_plan(epoch,round_id,permutations)
            rows = [None]*len(plan)
            identities = [tuple(map(int,ids)) for ids in plan]
            for shard,path in enumerate(self.paths):
                positions = np.flatnonzero(plan[:,0] == shard)
                if not len(positions):
                    continue
                table = pq.read_table(path, columns=['question','code'])
                for position,row in zip(positions,table.take(plan[positions,1]).to_pylist()):
                    rows[position] = row
            width = self.batch_size*self.window
            first_window = batch_offset//self.window
            for window_id in range(first_window, math.ceil(len(rows)/width)):
                start = window_id*width
                chunk = rows[start:start+width]
                source = self.tokenizer([r['question'] for r in chunk],
                    truncation=True, max_length=128)['input_ids']
                target = self.tokenizer([r['code'] for r in chunk],
                    truncation=True, max_length=self.target_length)['input_ids']
                ordered = (list(range(len(chunk))) if self.window == 1 else
                    sorted(range(len(chunk)), key=lambda i:(len(source[i]),min(192,len(target[i])))))
                groups = [ordered[i:i+self.batch_size] for i in range(0,len(ordered),self.batch_size)]
                if self.shuffle_batches:
                    self.rng(2,epoch,round_id,window_id).shuffle(groups)
                for group_id,group in enumerate(groups):
                    absolute_batch = window_id*self.window+group_id
                    if absolute_batch < batch_offset:
                        continue
                    next_cursor = [epoch,round_id,absolute_batch+1]
                    if next_cursor[2] == self.batches_in_round(round_id):
                        next_cursor = [epoch,round_id+1,0]
                        if next_cursor[1] == self.rounds:
                            next_cursor = [epoch+1,0,0]
                    batch = collate_tokens([source[i] for i in group], [target[i] for i in group],
                        pad_id=self.tokenizer.pad_token_id, eos_id=self.tokenizer.eos_token_id)
                    batch['_coverage'] = dict(cursor=list(cursor), next_cursor=next_cursor,
                        row_ids=[identities[start+i] for i in group])
                    yield batch
                    cursor = next_cursor

    def commit(self, ticket):
        """Called by the training consumer only after a successful optimizer update."""
        if ticket['cursor'] != self.state['cursor']:
            raise ValueError('Out-of-order or duplicated consumption receipt')
        ids = np.asarray(ticket['row_ids'], dtype='<i8')
        for shard,_ in ids:
            self.state['counts'][int(shard)] += 1
        self.state['presentations'] += len(ids)
        self.state['row_order_sha256'] = hashlib.sha256(
            bytes.fromhex(self.state['row_order_sha256'])+ids.tobytes()).hexdigest()
        if ticket['next_cursor'][0] > self.state['cursor'][0]:
            if self.state['counts'] != self.counts:
                raise ValueError('Epoch ended without visiting every eligible row')
            self.state['counts'] = [0]*len(self.counts)
        self.state['cursor'] = list(ticket['next_cursor'])

    def checkpoint(self):
        epoch = self.state['cursor'][0]
        unique = sum(self.counts) if epoch else sum(self.state['counts'])
        return dict(state=copy.deepcopy(self.state), report=dict(
            eligible_rows=sum(self.counts), eligible_by_shard=list(self.counts),
            excluded_by_shard=list(self.excluded),
            unique_visited_by_shard=list(self.counts if epoch else self.state['counts']),
            presentations_by_shard=[epoch*n+seen for n,seen in zip(self.counts,self.state['counts'])],
            visited_current_epoch_by_shard=list(self.state['counts']),
            completed_full_passes=epoch, unique_rows_since_tracking_started=unique,
            coverage_percent=100*unique/sum(self.counts),
            presentations=self.state['presentations'],
            repeated_presentations_since_tracking_started=self.state['presentations']-unique,
            historical_coverage_before_start_step='unknown',
            start_step=self.state['start_step'],
            source_length=128, target_length=self.target_length, grouping_target_length=192,
            note='Original row IDs, not distinct question texts; row coverage does not prove untruncated token coverage.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-shards', type=int, default=SHARDS)
    args = parser.parse_args()
    manifest = prepare_index(args.cache_dir, args.output_dir, args.max_shards)
    print(json.dumps({'eligible_rows':sum(r['eligible'] for r in manifest['shards']),
                      'manifest_sha256':sha256(args.output_dir/'manifest.json')}))
