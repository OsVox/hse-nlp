import pytest

from homeworks.hw2.adamw_size_curve import select
from homeworks.hw2.adamw_size_confirm import build_plan, paired_selection, LAUNCH
from homeworks.hw2.tests.test_adamw_size_curve import measurement


def prior(correct=11):
    return select({'adamw78': measurement(correct)},
                  dict(results=measurement(10), model_sha256='baseline'))


def test_no_confirmation_for_failed_candidate():
    with pytest.raises(ValueError, match='did not pass'):
        build_plan(prior(10))


def test_predeclared_seed_order_and_budget():
    plan = build_plan(prior())
    assert list(plan['arms']) == ['adamw78', 'adamw42']
    assert plan['options']['seed'] == 1042
    assert plan['options']['max_training_seconds'] == 600
    assert plan['options']['steps'] == 92456
    compile(LAUNCH, 'launch.py', 'exec')
    assert 'timeout=3160' in LAUNCH and 'runtime.unassign' in LAUNCH


def test_second_seed_requires_strict_gain_in_both_subsets():
    results = {'adamw42': measurement(10), 'adamw78': measurement(11)}
    result = paired_selection(results, prior())
    assert result['confirmed'] and result['full_pass_recipe'] == 'adamw78'
    assert result['screen_closed'] and not result['long_run_started']
    results['adamw78']['confirmation']['numeric_correct'] = 10
    assert paired_selection(results, prior())['full_pass_recipe'] == 'adamw42'


def test_second_seed_rejects_incompatible_measurements():
    results = {'adamw42': measurement(10), 'adamw78': measurement(11)}
    results['adamw78']['pilot']['max_input_tokens'] = 128
    with pytest.raises(ValueError, match='Incomparable'):
        paired_selection(results, prior())


def test_tampered_first_seed_choice_is_rejected():
    result = prior(10)
    result['provisional_winner'] = 'adamw78'
    with pytest.raises(ValueError, match='Prior selection'):
        build_plan(result)
