from copy import deepcopy
from pathlib import Path
import pytest

from homeworks.hw2.mature_growth_pilot import LAUNCH, check_budget, options_for, select
from homeworks.hw2.tests.test_adamw_size_curve import measurement
from homeworks.hw2.tokenizer_pilot import training_command


def test_resume_preserves_parent_architecture_and_schedule():
    options = options_for(dict(measured_training_seconds_per_arm=300,
        checkpoint_training_seconds=150), Path('/parent'), Path('/drive'), Path('/local'))
    command = training_command(options)
    assert '--resume-from' in command and '--predict-at-checkpoint' in command
    assert not any(flag in command for flag in ('--reset-optimizer', '--restart-schedule',
        '--d-ff', '--d-model', '--encoder-layers', '--decoder-layers'))
    assert options['steps'] == 184912 and options['learning_rate'] == 3e-5
    assert options['untruncated_training'] and options['prediction_max_input_tokens'] == 0


def test_launcher_has_bounded_release_and_offline_package_install():
    compile(LAUNCH, 'launch.py', 'exec')
    assert 'timeout=1800' in LAUNCH and 'runtime.unassign' in LAUNCH
    assert 'os.killpg' in LAUNCH and "ready['gpu_launch_ready']" in LAUNCH
    assert "'--no-index'" in LAUNCH and "'--find-links'" in LAUNCH
    assert "wheel_record['archive_sha256']" in LAUNCH and 'timeout=60' in LAUNCH
    assert 'homeworks.hw2.mature_growth_pilot' in LAUNCH


def test_only_consistent_two_subset_gain_passes():
    control = measurement(10)
    grown = measurement(11)
    passed = select(dict(control42=control, grown66=grown), control)
    assert passed['confirmation_required'] and not passed['long_run_started']
    grown['confirmation']['numeric_correct'] = 10
    assert select(dict(control42=control, grown66=grown), control)['provisional_winner'] == 'control42'
    grown = measurement(11, 4)
    assert not select(dict(control42=control, grown66=grown), control)['confirmation_required']
    grown['pilot']['questions_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Incomparable'):
        select(dict(control42=control, grown66=grown), control)


def budget_fixture():
    budget = dict(stop_reason='training_time_budget', training_seconds=300.1,
        checkpoint_steps=[123962, 124962], final_step=124962, updates=2000, examples_processed=256000)
    points = [dict(step=step, coverage=dict(presentations=1000000+128*updates),
        timing=dict(training_seconds=seconds, completed_updates=updates, examples_processed=128*updates))
        for step, updates, seconds in [(123962, 1000, 150.1), (124962, 2000, 300.1)]]
    return budget, points


def test_budget_accepts_complete_paired_intervals():
    budget, points = budget_fixture()
    check_budget(budget, points, 122962, 1000000)


@pytest.mark.parametrize('failure', ['timing', 'coverage', 'steps', 'updates'])
def test_budget_rejects_mislabeled_results(failure):
    budget, points = deepcopy(budget_fixture())
    if failure == 'timing': points[0]['timing']['training_seconds'] = 149
    if failure == 'coverage': points[1]['coverage']['presentations'] -= 128
    if failure == 'steps': budget['checkpoint_steps'][0] = budget['final_step']
    if failure == 'updates': budget['updates'] -= 1
    with pytest.raises(AssertionError):
        check_budget(budget, points, 122962, 1000000)
