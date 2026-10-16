from homeworks.hw2.learning_curve import PLAN, OPTIONS, CONFIGS, LAUNCH, select
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_bounded_full_pass_schedule_and_original_data():
    assert OPTIONS['steps']==92456 and OPTIONS['warmup']==500
    assert OPTIONS['max_training_seconds']==600 and OPTIONS['checkpoint_training_seconds']==300
    assert OPTIONS['untruncated_training'] and OPTIONS['coverage_index']
    assert OPTIONS['numeric_token_weight']==1 and OPTIONS['prediction_max_input_tokens']==0
    assert OPTIONS['prediction_max_new_tokens']==1280
    assert OPTIONS['encoder_layers']==6 and OPTIONS['decoder_layers']==4
    assert list(CONFIGS)==['adamw42','muon42','muon78']
    assert PLAN['outer_timeout_seconds']==3900
    compile(LAUNCH,'launch.py','exec')
    assert 'timeout=3900' in LAUNCH and 'homeworks.hw2.learning_curve' in LAUNCH
    assert 'runtime.unassign' in LAUNCH


def test_selects_large_only_with_consistent_size_gain():
    scores=dict(adamw42=result(20,30),muon42=result(25,35),muon78=result(24,40))
    assert select(scores)['provisional_winner']=='muon42'
    scores['muon78']=result(26,36)
    assert select(scores)['provisional_winner']=='muon78'
    scores['muon78']=result(26,36,1)
    assert select(scores)['provisional_winner']=='muon42'
    scores['muon42']=result(19,35)
    assert select(scores)['provisional_winner']=='adamw42'
    scores['muon78']=result(26,36)
    assert select(scores)['provisional_winner']=='muon78'
    assert not select(scores)['public_test_used_for_selection']
