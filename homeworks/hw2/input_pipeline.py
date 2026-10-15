"""Tokenize training windows once and optionally prepare batches ahead of CUDA.

The reference text, sampling stream, grouping order, truncation, stop-token rule
and loss mask match the original training path. No target text is rewritten.
"""
from itertools import islice
from queue import Empty, Full, Queue
from threading import Event, Thread

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_tokens(sources, targets, *, pad_id, eos_id):
    inputs = pad_sequence([torch.tensor(x, dtype=torch.long) for x in sources],
                          batch_first=True, padding_value=pad_id)
    labels = pad_sequence([torch.tensor(x, dtype=torch.long) for x in targets],
                          batch_first=True, padding_value=pad_id)
    # Match encode_batch, including literal pad tokens inside supplied text.
    lengths = labels.ne(pad_id).sum(dim=1)
    labels[torch.arange(len(targets)), lengths - 1] = eos_id
    labels[labels == pad_id] = -100
    source_lengths = torch.tensor([len(x) for x in sources])
    mask = (torch.arange(inputs.shape[1])[None, :] < source_lengths[:, None]).long()
    return {'input_ids': inputs, 'attention_mask': mask, 'labels': labels}


def encoded_batches(batches, tokenizer, batch_size, window):
    if window < 1 or batch_size < 1:
        raise ValueError('batch size and grouping window must be positive')
    if tokenizer.padding_side != 'right':
        raise ValueError('This model pipeline requires right padding')
    batches = iter(batches)
    while chunk := list(islice(batches, window)):
        rows = [row for batch in chunk for row in batch]
        if not rows:
            continue
        source = tokenizer([r['question'] for r in rows], truncation=True,
                           max_length=128)['input_ids']
        target = tokenizer([r['code'] for r in rows], truncation=True,
                           max_length=192)['input_ids']
        if window == 1:
            # The legacy path preserves an incoming partial batch in this case.
            groups = [list(range(len(rows)))]
        else:
            order = sorted(range(len(rows)), key=lambda i: (len(source[i]), len(target[i])))
            groups = (order[start:start + batch_size] for start in range(0, len(order), batch_size))
        for indices in groups:
            yield collate_tokens([source[i] for i in indices], [target[i] for i in indices],
                                 pad_id=tokenizer.pad_token_id, eos_id=tokenizer.eos_token_id)


def prefetch_batches(batches, depth=2, *, pin_memory=False):
    """Bounded CPU producer; pass errors to the consumer and stop on close."""
    if depth < 0:
        raise ValueError('prefetch depth must be non-negative')
    if depth == 0:
        for batch in batches:
            yield {k: v.pin_memory() for k, v in batch.items()} if pin_memory else batch
        return
    queue = Queue(maxsize=depth)
    stopped = Event()

    def put(kind, value):
        while not stopped.is_set():
            try:
                queue.put((kind, value), timeout=0.1)
                return
            except Full:
                pass

    def produce():
        try:
            for batch in batches:
                if stopped.is_set():
                    break
                if pin_memory:
                    batch = {k: v.pin_memory() for k, v in batch.items()}
                put('batch', batch)
        except BaseException as error:
            put('error', error)
        finally:
            put('end', None)

    producer = Thread(target=produce, name='hw2-input-prefetch', daemon=True)
    producer.start()
    try:
        while True:
            try:
                kind, value = queue.get(timeout=0.1)
            except Empty:
                if not producer.is_alive():
                    raise RuntimeError('Input producer exited without an end marker')
                continue
            if kind == 'end':
                return
            if kind == 'error':
                raise value
            yield value
    finally:
        stopped.set()
        producer.join(timeout=1)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=device.type == 'cuda')
            for key, value in batch.items()}
