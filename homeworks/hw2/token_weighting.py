"""Optional loss weighting of unchanged numeric/operator tokenizer pieces.

This is a training hypothesis, not a program repair or syntax-aware parser.
Pieces inside comments or strings also receive weight when they match.
"""
import math
import re
import torch
import torch.nn.functional as F


def token_weights(tokenizer, factor):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError('numeric token weight must be finite and at least one')
    weights = torch.ones(len(tokenizer), dtype=torch.float32)
    special = set(tokenizer.all_special_ids)
    for i in range(len(tokenizer)):
        piece = tokenizer.decode([i], clean_up_tokenization_spaces=False).strip()
        if i not in special and re.fullmatch(r'(?:[0-9]+(?:\.[0-9]+)?|[+*/%\-]+)', piece):
            weights[i] = factor
    return weights


def weighted_loss(logits, labels, weights, *, smoothing=0.0, example_balanced=False,
                  focal_gamma=0.0):
    """Normalize by valid-token weight; ignored padding adds no loss/gradient."""
    flat = labels.reshape(-1)
    valid = flat.ne(-100)
    per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                flat, reduction='none', ignore_index=-100, label_smoothing=smoothing)
    if not math.isfinite(focal_gamma) or focal_gamma < 0 or (focal_gamma and smoothing):
        raise ValueError('Focal gamma must be nonnegative and cannot combine with smoothing')
    if focal_gamma:
        # Standard focal modulation, including its gradient, with alpha=1.
        # The denominator stays the original valid-token weight, not focal weight.
        modulation = (-torch.expm1(-per_token)).clamp_min(torch.finfo(per_token.dtype).tiny)
        per_token = per_token * modulation.pow(focal_gamma)
    selected = weights[flat.clamp_min(0)] * valid
    if example_balanced:
        numerator = (per_token * selected).reshape(labels.shape).sum(dim=1)
        denominator = selected.reshape(labels.shape).sum(dim=1)
        usable = denominator.gt(0)
        return (numerator / denominator.clamp_min(1))[usable].mean()
    return (per_token * selected).sum() / selected.sum().clamp_min(1)
