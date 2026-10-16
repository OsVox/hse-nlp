"""Measured training budget and reproducible random state for resumable runs."""
import math
import random
import time

import numpy as np
import torch


class TrainingClock:
    """Count complete optimizer updates, including input waits, excluding exports."""
    def __init__(self, limit=None, checkpoint_every=None, *, now=time.monotonic):
        for value in (limit, checkpoint_every):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Time budgets must be finite and positive')
        if checkpoint_every is not None and limit is None:
            raise ValueError('Timed checkpoints require a training limit')
        self.limit, self.interval, self.now = limit, checkpoint_every, now
        self.elapsed = 0.
        self.started = None
        self.next_checkpoint = checkpoint_every

    def begin_update(self):
        if self.started is not None:
            raise RuntimeError('Update timer already running')
        self.started = self.now()

    def end_update(self):
        if self.started is None:
            raise RuntimeError('Update timer is not running')
        self.elapsed += self.now() - self.started
        self.started = None
        due = self.next_checkpoint is not None and self.elapsed >= self.next_checkpoint
        if due:
            self.next_checkpoint = (math.floor(self.elapsed/self.interval)+1)*self.interval
        return due, self.limit is not None and self.elapsed >= self.limit


def capture_rng():
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    cuda = state['torch_cuda']
    if cuda and len(cuda) != torch.cuda.device_count():
        raise ValueError('Exact RNG resume requires the same CUDA device count')
    torch.set_rng_state(state['torch_cpu'])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
