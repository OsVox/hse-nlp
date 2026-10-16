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


def weighted_loss(logits, labels, weights):
    """Normalize by valid-token weight; ignored padding adds no loss/gradient."""
    flat = labels.reshape(-1)
    valid = flat.ne(-100)
    per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                flat, reduction='none', ignore_index=-100)
    selected = weights[flat.clamp_min(0)] * valid
    return (per_token * selected).sum() / selected.sum().clamp_min(1)
