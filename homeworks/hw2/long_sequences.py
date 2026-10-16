"""Split a logical CPU batch without changing row order or its loss reduction.

Budgets bound padded tokens and attention positions, not measured GPU bytes.
An oversized single example raises before transfer; it is never silently cut.
"""
import torch


def split_batch(batch, *, token_budget, attention_budget, weights=None,
                example_balanced=False):
    if min(token_budget, attention_budget) < 1:
        raise ValueError('Microbatch budgets must be positive')
    if any(v.device.type != 'cpu' for v in batch.values()):
        raise ValueError('Split batches on CPU before GPU transfer')
    labels = batch['labels']
    source_lengths = batch['attention_mask'].sum(1).tolist()
    target_lengths = labels.ne(-100).sum(1).tolist()
    if not source_lengths or min(source_lengths + target_lengths) < 1:
        raise ValueError('Empty sequences are not supported')
    valid = labels.ne(-100)
    if example_balanced:
        mass = valid.any(1).double()
    elif weights is None:
        mass = valid.sum(1).double()
    else:
        if weights.device.type != 'cpu':
            raise ValueError('Loss weights must be on CPU')
        mass = (weights[labels.clamp_min(0)].double() * valid).sum(1)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError('Invalid loss normalization mass')

    def fits(n, s, t):
        return n*(s+t) <= token_budget and n*(s*s+t*t+s*t) <= attention_budget

    # Validate every row before yielding anything, avoiding a partial backward
    # followed by an over-budget exception on a later row.
    for s,t in zip(source_lengths,target_lengths):
        if not fits(1,s,t):
            raise ValueError(f'Single example exceeds microbatch budget: source={s}, target={t}')
    total = mass.sum().item()
    start = 0
    while start < len(labels):
        end, smax, tmax = start, 0, 0
        while end < len(labels):
            s, t = max(smax,source_lengths[end]), max(tmax,target_lengths[end])
            if not fits(end-start+1,s,t):
                break
            end, smax, tmax = end+1, s, t
        part = dict(input_ids=batch['input_ids'][start:end,:smax],
                    attention_mask=batch['attention_mask'][start:end,:smax],
                    labels=labels[start:end,:tmax])
        yield part, mass[start:end].sum().item()/total
        start = end
