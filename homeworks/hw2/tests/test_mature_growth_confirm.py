from copy import deepcopy
import random
import numpy as np
import pytest
import torch

from homeworks.hw2.mature_growth_confirm import LAUNCH, select
from homeworks.hw2.tests.test_adamw_size_curve import measurement
from homeworks.hw2.training_budget import capture_rng, restore_rng, seed_continuation


def test_both_repetitions_required_and_mixed_results_rejected():
    control, grown = measurement(10), measurement(11)
    prior = dict(results=dict(control42=control, grown66=grown))
    assert select(prior['results'], control, prior)['scale_recipe'] == 'grown66'
    mixed = deepcopy(prior['results'])
    mixed['grown66']['confirmation']['numeric_correct'] = 10
    assert select(mixed, control, prior)['scale_recipe'] is None
    mixed['grown66']['confirmation']['questions_sha256'] = 'different'
    with pytest.raises(ValueError):
        select(mixed, control, prior)
    with pytest.raises(ValueError):
        select(prior['results'], control, dict(results=dict(control42=grown, grown66=control)))


def test_explicit_continuation_rng_is_repeatable_and_changes_dropout():
    original = capture_rng()
    try:
        def draw(seed):
            seed_continuation(seed)
            return torch.nn.functional.dropout(torch.ones(1000), .5), random.random(), np.random.rand()
        a, b, c = draw(42), draw(1042), draw(1042)
        assert not torch.equal(a[0], b[0])
        assert torch.equal(b[0], c[0]) and b[1:] == c[1:]
    finally:
        restore_rng(original)


def test_confirmation_launcher_keeps_offline_install_timeout_and_release():
    compile(LAUNCH, 'confirmation-launch.py', 'exec')
    assert 'homeworks.hw2.mature_growth_confirm' in LAUNCH
    assert "'--no-index'" in LAUNCH and 'timeout=1800' in LAUNCH
    assert 'runtime.unassign' in LAUNCH
