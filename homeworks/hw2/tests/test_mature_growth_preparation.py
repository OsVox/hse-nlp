from copy import deepcopy
import pytest

from homeworks.hw2.prepare_mature_growth import plan_from_parent


def parent_config():
    return dict(optimizer='adamw', seed=42, steps=184912, warmup=0,
        learning_rate=3e-5, schedule_start=92456, batch_size=128, accumulation=1,
        untruncated_training=True, group_window=32, input_pipeline='encoded',
        prefetch_batches=2, microbatch_token_budget=40960,
        microbatch_attention_budget=9961472,
        model_config=dict(d_model=512, d_ff=1536, num_heads=8, num_layers=6,
            num_decoder_layers=4, vocab_size=8192, feed_forward_proj='gated-gelu'))


def test_preparation_is_bounded_and_does_not_authorize_a_launch():
    config=parent_config(); original=deepcopy(config)
    plan=plan_from_parent(config,122962,'own-model')
    assert config==original
    assert plan['total_measured_training_seconds']==600
    assert plan['measured_training_seconds_per_arm']==300
    assert plan['checkpoint_training_seconds']==150
    assert not plan['gpu_launch_ready']
    assert not plan['reset_optimizer'] and not plan['restart_schedule']
    assert plan['preserve_data_cursor'] and plan['full_sequences']
    assert not plan['public_test_used_for_selection']


@pytest.mark.parametrize('key,value',[
    ('optimizer','muon'),('seed',43),('steps',200000),('learning_rate',.0003),
    ('schedule_start',122962),('batch_size',256),('untruncated_training',False)])
def test_changed_trajectory_rejected(key,value):
    config=parent_config();config[key]=value
    with pytest.raises(ValueError,match='parent setting'):
        plan_from_parent(config,122962,'own-model')


def test_public_selected_parent_and_changed_architecture_rejected():
    with pytest.raises(ValueError,match='held-out parent'):
        plan_from_parent(parent_config(),153541,'public-best')
    config=parent_config();config['model_config']['num_decoder_layers']=6
    with pytest.raises(ValueError,match='architecture'):
        plan_from_parent(config,122962,'different-model')
