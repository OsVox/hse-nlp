import torch
from homeworks.hw2.size_pilot import PLAN, CONFIGS, LAUNCH, select
from homeworks.hw2.train import new_model
from homeworks.hw2.tests.test_optimizer_pilot import result


def test_capacity_and_attention_dimensions():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        def __len__(self): return 8192
    for name, config in CONFIGS.items():
        with torch.device('meta'):
            model = new_model(Tokenizer(), encoder_layers=6, decoder_layers=4, **config)
        assert sum(p.numel() for p in model.parameters()) == PLAN['expected_parameters'][name]
        assert model.config.d_kv == 64
        assert model.config.num_layers == 6 and model.config.num_decoder_layers == 4
    assert PLAN['save_every'] == PLAN['steps'] == 2048
    assert PLAN['eval_every'] == 512
    compile(LAUNCH, 'launch.py', 'exec')
    assert 'timeout=1400' in LAUNCH and 'RTX PRO 6000' in LAUNCH


def test_shortlist_requires_two_gains_and_no_extra_failures():
    r = select(dict(control=result(10, 20), small=result(11, 21), large=result(12, 19)))
    assert r['shortlist'] == ['small']
    assert not r['equal_gpu_time'] and not r['optimal_model_size_established']
    r = select(dict(control=result(10, 20), small=result(11, 21, 1), large=result(10, 20)))
    assert not r['shortlist'] and not r['long_run_started']
