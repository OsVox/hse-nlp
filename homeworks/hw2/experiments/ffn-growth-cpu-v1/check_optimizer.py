"""Check growth using actual saved AdamW history; discard the CPU update."""
import hashlib
import json
from pathlib import Path
import signal
import time

import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from homeworks.hw2.grow_ffn import grow_feed_forward, grow_adamw_optimizer


signal.alarm(300)
torch.set_num_threads(2)
assert not torch.cuda.is_available()
directory = Path('/Users/andrejvorsin/Downloads/hw2-selected-scale-v1/step-122962')
manifest = json.loads((directory/'checkpoint.complete.json').read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['model.safetensors', 'training_state.pt']:
    assert sha(directory/name) == manifest['files'][name]
started = time.monotonic()
parent = T5ForConditionalGeneration.from_pretrained(directory, local_files_only=True).eval()
state = torch.load(directory/'training_state.pt', map_location='cpu', weights_only=False)
assert state['step'] == 122962
optimizer = torch.optim.AdamW(parent.parameters(), lr=3e-5, weight_decay=.01)
optimizer.load_state_dict(state['optimizer'])
grown = grow_feed_forward(parent, 3072).eval()
new_optimizer = grow_adamw_optimizer(parent, grown, optimizer)
old_named, new_named = dict(parent.named_parameters()), dict(grown.named_parameters())
resized = 0
for name, old in old_named.items():
    new = new_named[name]
    old_state, new_state = optimizer.state[old], new_optimizer.state[new]
    assert old_state['step'].item() == new_state['step'].item() == 122962
    for key in ['exp_avg', 'exp_avg_sq']:
        slices = tuple(slice(0, dim) for dim in old.shape)
        assert torch.equal(old_state[key], new_state[key][slices])
        assert old_state[key].data_ptr() != new_state[key].data_ptr()
        if old.shape != new.shape:
            rest = new_state[key].clone(); rest[slices] = 0
            assert torch.count_nonzero(rest) == 0
    resized += old.shape != new.shape
assert resized == 30
data = Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset/part-00001.parquet')
rows = next(pq.ParquetFile(data).iter_batches(batch_size=4, columns=['question', 'code'])).to_pylist()
tokenizer = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True)
inputs = tokenizer([r['question'] for r in rows], padding=True, truncation=False,
    return_tensors='pt', return_token_type_ids=False)
targets = tokenizer([r['code'] for r in rows], padding=True, truncation=False,
    return_tensors='pt', return_token_type_ids=False)
labels = targets['input_ids'].masked_fill(targets['attention_mask'] == 0, -100)
with torch.no_grad():
    before = parent(**inputs, labels=labels).logits
    after = grown(**inputs, labels=labels).logits
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)
loss = grown(**inputs, labels=labels).loss
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in grown.parameters() if p.grad is not None)
layer = grown.encoder.block[0].layer[-1].DenseReluDense
assert layer.wo.weight.grad[:, 1536:].abs().sum() > 0
new_optimizer.step()
assert all(torch.isfinite(p).all() for p in grown.parameters())
assert all(s['step'].item() == 122963 for s in new_optimizer.state.values())
assert all(s['step'].item() == 122962 for s in optimizer.state.values())
for name in ['model.safetensors', 'training_state.pt']:
    assert sha(directory/name) == manifest['files'][name]
record = dict(status='passed', parent_step=122962, cpu_probe_step=122963,
    parent_model_sha256=manifest['files']['model.safetensors'],
    parent_training_state_sha256=manifest['files']['training_state.pt'],
    original_state_unchanged=True, resized_parameter_tensors=resized,
    original_moments_equal=True, added_moments_initially_zero=True,
    preserved_parameter_step=True, learning_rate=new_optimizer.param_groups[0]['lr'],
    max_absolute_logit_difference=(before-after).abs().max().item(), finite_update=True,
    original_examples=4, row_indices=[0,1,2,3], data_sha256=sha(data),
    elapsed_seconds=time.monotonic()-started, torch_version=torch.__version__,
    gpu=False, retained_probe_weights=False, quality_gain_verified=False,
    policy='Keep each tensor step at the parent count; new coordinates share its mature bias correction',
    limitations=['CPU eval-mode finite update only, not a training quality experiment',
        'Scheduler, RNG and data-cursor migration were not exercised by this helper',
        'A mature tensor step gives new coordinates different bias correction from fresh AdamW'])
Path('homeworks/hw2/experiments/ffn-growth-cpu-v1/optimizer-verification.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
