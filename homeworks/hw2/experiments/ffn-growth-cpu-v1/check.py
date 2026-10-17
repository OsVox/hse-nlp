"""Check weight-preserving growth on the real parent; no training or GPU."""
import hashlib
import json
from pathlib import Path
import signal
import time

import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from homeworks.hw2.grow_ffn import grow_feed_forward


signal.alarm(300)
torch.set_num_threads(2)
assert not torch.cuda.is_available()
parent_dir = Path('/Users/andrejvorsin/Downloads/hw2-selected-scale-v1/step-102563')
weights = parent_dir/'model.safetensors'
expected = 'c758cd0d3ba81806098867b63fcae44789b0e1ad8e9d8b8a3dcecb9f50a27441'
assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected
parent = T5ForConditionalGeneration.from_pretrained(parent_dir, local_files_only=True).eval()
tokenizer = PreTrainedTokenizerFast.from_pretrained(parent_dir, local_files_only=True)
data = Path('/Users/andrejvorsin/Downloads/hw2-full-coverage/dataset/part-00001.parquet')
rows = next(pq.ParquetFile(data).iter_batches(batch_size=4, columns=['question', 'code'])).to_pylist()
inputs = tokenizer([r['question'] for r in rows], padding=True, truncation=False,
    return_tensors='pt', return_token_type_ids=False)
targets = tokenizer([r['code'] for r in rows], padding=True, truncation=False,
    return_tensors='pt', return_token_type_ids=False)
labels = targets['input_ids'].masked_fill(targets['attention_mask'] == 0, -100)
started = time.monotonic()
grown = grow_feed_forward(parent, 3072, seed=42).eval()
with torch.inference_mode():
    baseline = parent(**inputs, labels=labels).logits
    candidate = grown(**inputs, labels=labels).logits
    delta = (candidate-baseline).abs().max().item()
    torch.testing.assert_close(candidate, baseline, atol=1e-4, rtol=1e-4)
    original_ids = parent.generate(**inputs, max_new_tokens=32, do_sample=False)
    cached_ids = grown.generate(**inputs, max_new_tokens=32, do_sample=False)
    uncached_ids = grown.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=False)
    assert torch.equal(original_ids, cached_ids) and torch.equal(original_ids, uncached_ids)
assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected
record = dict(status='passed', model_sha256=expected, parent_parameters=sum(p.numel() for p in parent.parameters()),
    grown_parameters=sum(p.numel() for p in grown.parameters()), original_d_ff=1536, grown_d_ff=3072,
    examples=4, data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
    original_row_indices=[0,1,2,3], input_shape=list(inputs['input_ids'].shape),
    target_shape=list(labels.shape), max_absolute_logit_difference=delta,
    cached_and_uncached_greedy_ids_equal=True, greedy_test_max_new_tokens=32,
    elapsed_seconds=time.monotonic()-started, torch_version=torch.__version__,
    training=False, gpu=False, quality_gain_verified=False,
    limitations=['Four-example numerical check is not whole-corpus equivalence',
        'No optimizer migration, GPU throughput or quality pilot has been run',
        'Dropout training masks are not guaranteed identical after widening'])
out = Path('homeworks/hw2/experiments/ffn-growth-cpu-v1/verification.json')
out.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
