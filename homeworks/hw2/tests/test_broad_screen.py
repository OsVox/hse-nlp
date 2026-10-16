"""Synthetic assertions only; these examples never enter training."""
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from homeworks.hw2.token_weighting import weighted_loss
from homeworks.hw2.train import new_model, override_dropout
from homeworks.hw2.input_pipeline import encoded_batches
from homeworks.hw2.broad_screen import select_candidates


def test_smoothing_and_equal_question_loss():
    torch.manual_seed(5)
    logits=torch.randn(3,4,5,requires_grad=True)
    labels=torch.tensor([[1,2,3,4],[2,-100,-100,-100],[-100,-100,-100,-100]])
    expected=torch.stack([F.cross_entropy(logits[i],labels[i],label_smoothing=.05) for i in range(2)]).mean()
    loss=weighted_loss(logits,labels,torch.ones(5),smoothing=.05,example_balanced=True)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert logits.grad[labels.eq(-100)].count_nonzero()==0


def test_dropout_applies_to_attention_and_survives_save(tmp_path):
    tokenizer=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    model=new_model(tokenizer,d_model=64,d_ff=128,encoder_layers=1,decoder_layers=1,num_heads=1)
    override_dropout(model,.05)
    assert all(m.p==.05 for m in model.modules() if isinstance(m,torch.nn.Dropout))
    assert all(m.dropout==.05 for m in model.modules() if isinstance(getattr(m,'dropout',None),float))
    model.save_pretrained(tmp_path)
    restored=type(model).from_pretrained(tmp_path)
    assert all(m.p==.05 for m in restored.modules() if isinstance(m,torch.nn.Dropout))


def test_filter_drops_incomplete_targets_but_retains_full_batches():
    tokenizer=PreTrainedTokenizerFast.from_pretrained('homeworks/hw2/tokenizer')
    rows=[{'question':str(i),'code':('x = 1\n'*400 if i%3==0 else 'return 1')} for i in range(15)]
    result=list(encoded_batches([rows[i:i+2] for i in range(0,len(rows),2)],tokenizer,2,2,skip_truncated_targets=True))
    assert len(result)==5
    assert all(b['labels'].shape[0]==2 for b in result)
    assert all(b['labels'].shape[1]<192 for b in result)


def test_shortlist_rejects_split_regressions_and_execution_regressions():
    def arm(a,b,f=1):return {'pilot':{'numeric_correct':a,'prediction_failures':f},'confirmation':{'numeric_correct':b,'prediction_failures':f}}
    result=select_candidates({'control':arm(10,20),'winner':arm(11,22),'mixed':arm(9,25),'unsafe':arm(12,25,2),'tie':arm(10,20)})
    assert result['shortlist']==['winner']
    assert not result['long_run_authorized_by_evidence']
