import random
import numpy as np
import pytest
import torch
from homeworks.hw2.training_budget import TrainingClock, capture_rng, restore_rng


def test_only_completed_updates_consume_budget_and_exports_do_not():
    t=[0.]
    clock=TrainingClock(3.,1.,now=lambda:t[0])
    clock.begin_update();t[0]=1.2
    assert clock.end_update()==(True,False)
    t[0]=100.  # Model/export/evaluation time is outside training.
    clock.begin_update();t[0]=100.5
    assert clock.end_update()==(False,False)
    clock.begin_update();t[0]=102.
    assert clock.end_update()==(True,True)
    assert clock.elapsed==pytest.approx(3.2)


def test_invalid_budget_and_incomplete_update():
    for v in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):TrainingClock(v)
    with pytest.raises(ValueError):TrainingClock(checkpoint_every=1)
    clock=TrainingClock()
    with pytest.raises(RuntimeError):clock.end_update()
    clock.begin_update()
    with pytest.raises(RuntimeError):clock.begin_update()


def test_rng_restore_replays_dropout_and_other_random_draws():
    original=capture_rng()
    try:
        state=capture_rng()
        first=(torch.nn.functional.dropout(torch.ones(100),.5),random.random(),np.random.rand(10))
        restore_rng(state)
        second=(torch.nn.functional.dropout(torch.ones(100),.5),random.random(),np.random.rand(10))
        assert torch.equal(first[0],second[0])
        assert first[1]==second[1]
        np.testing.assert_array_equal(first[2],second[2])
    finally:restore_rng(original)


def test_timed_training_saves_actual_step_and_resumes_full_schedule(tmp_path, monkeypatch):
    import json
    import sys
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    from homeworks.hw2 import train
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    vocab={'<pad>':0,'</s>':1,'<unk>':2,'question':3,'def':4,'solve':5,'return':6,'1':7,':':8,'(':9,')':10}
    backend=Tokenizer(models.WordLevel(vocab,unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    backend.post_processor=processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='</s>',unk_token='<unk>')
    token_dir=tmp_path/'tokenizer';tokenizer.save_pretrained(token_dir)
    row=dict(question='question',code='def solve(): return 1')
    def batches(*args,**kwargs):
        while True:yield [row]
    monkeypatch.setattr(train,'training_batches',batches)
    monkeypatch.setattr(train,'validation_examples',lambda cache:[row])
    out=tmp_path/'training'
    common=['train','--steps','999','--max-training-seconds','0.000001',
        '--batch-size','1','--accumulation','1','--group-window','1','--precision','fp32',
        '--save-every','999','--eval-every','999','--output-dir',str(out),'--warmup','0']
    monkeypatch.setattr(sys,'argv',common+['--tokenizer-dir',str(token_dir),
        '--d-model','64','--d-ff','128','--encoder-layers','1','--decoder-layers','1','--num-heads','1'])
    train.main()
    first=json.loads((out/'compute-budget.json').read_text())
    assert first['final_step']==first['updates']==1
    assert first['planned_final_step']==999 and first['checkpoint_steps']==[1]
    assert first['stop_reason']=='training_time_budget'
    timing=json.loads((out/'latest/training-budget.json').read_text())
    assert timing['optimizer_step']==1 and timing['time_limit_reached']
    assert timing['training_seconds']==first['training_seconds']
    state=torch.load(out/'latest/training_state.pt',weights_only=False)
    assert state['step']==1 and 'rng' in state
    assert state['optimizer']['param_groups'][0]['lr']>.00029
    monkeypatch.setattr(sys,'argv',common)
    train.main()
    second=json.loads((out/'compute-budget.json').read_text())
    assert second['final_step']==2 and second['updates']==1
    assert second['checkpoint_steps']==[2]
    state=torch.load(out/'latest/training_state.pt',weights_only=False)
    assert state['scheduler']['last_epoch']==2
    # The interrupted run must produce the same weights as two uninterrupted
    # updates, including dropout draws and optimizer moments.
    from safetensors.torch import load_file
    resumed_weights=load_file(out/'latest/model.safetensors')
    ticks=iter([0.,1.,2.,3.])
    monkeypatch.setattr(train,'TrainingClock',lambda *args:TrainingClock(2.,now=lambda:next(ticks)))
    reference=tmp_path/'uninterrupted'
    monkeypatch.setattr(sys,'argv',common+['--output-dir',str(reference),'--tokenizer-dir',str(token_dir),
        '--d-model','64','--d-ff','128','--encoder-layers','1','--decoder-layers','1','--num-heads','1'])
    train.main()
    reference_weights=load_file(reference/'latest/model.safetensors')
    assert all(torch.equal(resumed_weights[k],reference_weights[k]) for k in resumed_weights)
    monkeypatch.setattr(sys,'argv',common+['--steps','1000'])
    with pytest.raises(SystemExit):train.main()
