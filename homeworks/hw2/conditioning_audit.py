"""CPU diagnostic of question and preceding-code conditioning; no training.

All loss labels are untouched supplied training programs on the reserved first
256 rows. Question permutation and embedding removal are evaluation ablations,
not training examples. Their effects are not directly comparable causal shares.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .checkpoint_io import sha256
from .data import validation_examples
from .token_weighting import token_weights
from .train import encode_batch


def summarize(logits, labels, numeric):
    valid = labels.ne(-100)
    loss = F.cross_entropy(logits.float().transpose(1, 2), labels,
                           reduction='none', ignore_index=-100)
    correct = logits.argmax(-1).eq(labels)
    numeric_mask = numeric[labels.clamp_min(0)] & valid
    masks = {'all': valid, 'numeric_operator': numeric_mask,
             'other': valid & ~numeric_mask}
    return {name: {'tokens': int(mask.sum()),
                   'loss_sum': float(loss[mask].sum()),
                   'correct': int(correct[mask].sum())}
            for name, mask in masks.items()}


def run(checkpoint, cache, output, source_commit, rows=256, batch_size=8):
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    assert not output.exists(), 'Use a fresh diagnostic destination'
    assert rows == 256, 'Only the existing loss-validation subset is authorized here'
    expected = 'cad272fa2644c3876a577affc6f8559be9d5efce4e0f5c66f8f3ea6b7fb84cb6'
    manifest = json.loads((checkpoint/'checkpoint.complete.json').read_text())
    for name in ['model.safetensors', 'config.json', 'tokenizer.json']:
        assert sha256(checkpoint/name) == manifest['files'][name]
    assert sha256(checkpoint/'model.safetensors') == expected
    torch.set_num_threads(4)
    began = time.monotonic()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).float().cpu().eval()
    assert next(model.parameters()).device.type == 'cpu'
    examples = validation_examples(cache)[:rows]
    numeric = token_weights(tokenizer, 2).gt(1)
    totals = {arm: {} for arm in ['original', 'permuted_question', 'no_code_prefix']}
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            items = examples[start:start+batch_size]
            batch = encode_batch(items, tokenizer, torch.device('cpu'))
            shifted = [{'question': examples[(i+rows//2) % rows]['question'],
                        'code': examples[i]['code']}
                       for i in range(start, start+len(items))]
            permuted = encode_batch(shifted, tokenizer, torch.device('cpu'))
            assert torch.equal(batch['labels'], permuted['labels'])
            decoder_ids = model.prepare_decoder_input_ids_from_labels(batch['labels'])
            embeddings = model.get_input_embeddings()(decoder_ids).clone()
            # Preserve the start embedding, remove all preceding-code content.
            embeddings[:, 1:] = 0
            conditions = {'original': batch, 'permuted_question': permuted,
                          'no_code_prefix': {**batch, 'decoder_inputs_embeds': embeddings}}
            for arm, inputs in conditions.items():
                result = summarize(model(**inputs, use_cache=False).logits, batch['labels'], numeric)
                for group, counts in result.items():
                    dest = totals[arm].setdefault(group, dict(tokens=0, loss_sum=0., correct=0))
                    for key, value in counts.items():
                        dest[key] += value
            print('AUDIT_ROWS', start+len(items), rows, flush=True)
    for groups in totals.values():
        for counts in groups.values():
            counts['mean_loss'] = counts['loss_sum']/counts['tokens']
            counts['accuracy'] = counts['correct']/counts['tokens']
    result = dict(source_commit=source_commit, model_sha256=expected, rows=rows,
                  data='Original train shard 0, reserved rows 0:256; no public test examples',
                  examples_sha256=hashlib.sha256(json.dumps(examples,ensure_ascii=False).encode()).hexdigest(),
                  shard_sha256=sha256(cache/'part-00000.parquet'),
                  device='cpu', training=False, precision='float32', batch_size=batch_size,
                  seconds=time.monotonic()-began, conditions=totals,
                  limitation='Teacher-forced token diagnostics are not program correctness. Unequal interventions do not estimate causal contribution shares or prove that dropout will help.')
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    a = p.parse_args()
    run(a.checkpoint, a.cache, a.output, a.source_commit)
