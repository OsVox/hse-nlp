from copy import deepcopy
import pytest

from homeworks.hw2.scaled_training import build_plan, validation_update, launcher
from homeworks.hw2.adamw_size_confirm import paired_selection
from homeworks.hw2.tests.test_adamw_size_confirm import prior
from homeworks.hw2.tests.test_adamw_size_curve import measurement


def config(width=512):
    return dict(optimizer='adamw',d_model=width,seed=42,untruncated_training=True,
        batch_size=128,accumulation=1,encoder_layers=6,decoder_layers=4,
        group_window=32,input_pipeline='encoded',prefetch_batches=2,
        microbatch_token_budget=40960,microbatch_attention_budget=9961472,
        learning_rate=.0003,warmup=500,steps=92456)


def test_reference_continues_with_explicit_low_lr_phase():
    plan=build_plan(prior(10),None,config(),92456,.1)
    assert plan['final_step']==184912 and plan['restart_first_schedule']
    assert plan['options']['learning_rate']==.00003 and plan['options']['warmup']==0
    assert 'initialization_vocab_size' not in plan['options']
    assert plan['options']['max_training_seconds']==1000
    assert plan['patience']==3
    compile(launcher(plan),'launch.py','exec')
    assert 'runtime.unassign' in launcher(plan)


def test_larger_model_needs_confirmation_and_preserves_schedule():
    with pytest.raises(ValueError,match='requires paired'):
        build_plan(prior(),None,config(704),4200,.14)
    confirmation=paired_selection({'adamw42':measurement(10),'adamw78':measurement(11)},prior())
    plan=build_plan(prior(),confirmation,config(704),4200,.14)
    assert plan['final_step']==92456 and not plan['restart_first_schedule']
    assert plan['options']['learning_rate']==.0003 and plan['options']['warmup']==500


def test_wrong_parent_or_modified_recipe_rejected():
    with pytest.raises(ValueError,match='completed first-pass'):
        build_plan(prior(10),None,config(),90000,.1)
    changed=config();changed['untruncated_training']=False
    with pytest.raises(ValueError,match='Unexpected parent'):
        build_plan(prior(10),None,changed,92456,.1)


def test_patience_counts_only_consecutive_non_improvements():
    state=dict(best=measurement(10),best_model_sha256='parent',stale_intervals=0)
    original=deepcopy(state)
    for _ in range(2):
        state,decision=validation_update(state,measurement(10),patience=3)
        assert not decision['stop']
    good=measurement(11)
    for value in good.values():value['model_sha256']='improved'
    state,decision=validation_update(state,good,patience=3)
    assert decision['improved'] and state['stale_intervals']==0
    assert original['stale_intervals']==0
    for _ in range(3):state,decision=validation_update(state,good,patience=3)
    assert decision['stop'] and state['best_model_sha256']=='improved'


def test_incomparable_scores_do_not_advance_stopping_state():
    state=dict(best=measurement(10),best_model_sha256='parent',stale_intervals=0)
    changed=measurement(11);changed['pilot']['questions_sha256']='different'
    with pytest.raises(ValueError,match='Incomparable'):
        validation_update(state,changed,patience=3)
    assert state['stale_intervals']==0
