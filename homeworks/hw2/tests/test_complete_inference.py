import csv
import json
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from homeworks.hw2 import inference


def test_complete_prompt_order_and_cap_hits_ignore_padded_finished_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    backend=Tokenizer(models.WordLevel({'<pad>':0,'</s>':1,'<unk>':2,'question':3,'x':4},unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    backend.post_processor=processors.TemplateProcessing(single='$A </s>',special_tokens=[('</s>',1)])
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='</s>',unk_token='<unk>')
    data=tmp_path/'test.parquet'
    questions=['question '*200,'question']
    pq.write_table(pa.table({'question':questions}),data)
    monkeypatch.setattr(inference,'fetch_parquet',lambda *args:data)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.lengths=[]
        def generate(self,**kwargs):
            self.lengths=kwargs['attention_mask'].sum(1).tolist()
            assert kwargs['max_new_tokens']==4 and kwargs['num_beams']==1 and not kwargs['do_sample']
            return torch.tensor([[0,4,1,0,0],[0,4,4,4,4]])
    model=Model();(tmp_path/'model.safetensors').write_bytes(b'test fixture only')
    output=tmp_path/'predictions.csv'
    inference.generate_predictions(tmp_path,output,tmp_path,model=model,tokenizer=tok,
                                   batch_size=2,max_input_tokens=0,max_new_tokens=4)
    assert model.lengths==[2,201]
    with output.open() as f:assert [r['code'] for r in csv.DictReader(f)]==['x x x x','x']
    meta=json.loads(output.with_suffix('.provenance.json').read_text())
    assert meta['generation_limit_hits']==1 and meta['max_input_tokens'] is None
    inference.generate_predictions(tmp_path,output,tmp_path,model=model,tokenizer=tok,
                                   batch_size=2,max_input_tokens=128,max_new_tokens=4)
    assert model.lengths==[2,128]
