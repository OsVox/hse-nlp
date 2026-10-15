"""Measure our existing train-only probe and count candidate T5 parameters.

The probe samples about 256 rows per shard, so corpus totals are approximate
(the final shard is smaller). They are not deduplicated token counts. Chinchilla
is a decoder-only, compute-optimal cross-entropy result, not a pass@1 predictor
or an encoder-decoder sizing formula. Use D/20 only to choose a pilot range.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration


def estimate(pool, tokenizer_path, train_rows):
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    result = dict(sample_rows=len(rows),unique_questions_in_sample=len({r['question'] for r in rows}),
                  train_rows=train_rows,probe_sha256=digest,
                  sampling_note='Approximately 256 rows per shard; unequal shard sizes; '
                  'token totals are approximate and not deduplicated.',
                  tokenizer_sha256=hashlib.sha256((tokenizer_path/'tokenizer.json').read_bytes()).hexdigest())
    for key, cap in [('question',128),('code',192)]:
        lengths = [len(x) for x in tokenizer([r[key] for r in rows],truncation=False)['input_ids']]
        mean_capped = statistics.mean(min(n,cap) for n in lengths)
        result[key] = dict(mean=statistics.mean(lengths),mean_capped=mean_capped,
                           truncated_fraction=sum(n>cap for n in lengths)/len(lengths),
                           estimated_capped_corpus_tokens=train_rows*mean_capped)
    result['architectures'] = []
    for width, encoder, decoder in [(256,4,4),(384,6,6),(512,4,4),(512,6,4),(512,6,6),(512,8,4)]:
        config = T5Config.from_pretrained(tokenizer_path)
        config.d_model=width; config.d_ff=3*width; config.num_layers=encoder
        config.num_decoder_layers=decoder; config.num_heads=width//64
        with torch.device('meta'):
            model = T5ForConditionalGeneration(config)
        result['architectures'].append(dict(width=width,encoder_layers=encoder,
            decoder_layers=decoder,parameters=sum(p.numel() for p in model.parameters())))
    target = result['code']['estimated_capped_corpus_tokens']
    total = target+result['question']['estimated_capped_corpus_tokens']
    result['heuristic'] = dict(target_tokens_divided_by_20=target/20,
        combined_tokens_divided_by_20=total/20,
        interpretation='Two rough token-accounting proxies, not a confidence interval or '
        'a fitted encoder-decoder optimum. Pilot sizes require held-out quality and actual T4 timing.',
        sources=['https://arxiv.org/abs/2203.15556','https://arxiv.org/abs/2109.07740'],
        proposed_first_candidate_parameters=42481152)
    result['training_time_projection'] = dict(
        assumption='Same sequence lengths and efficiency; seconds scale linearly with total '
        'parameters. This is a planning estimate, not measured larger-model throughput.',
        baseline_parameters=9967360,
        baseline_examples_per_second=[3692.8,3200],
        baseline_note='3692.8 measured short matched probe; 3200 rounded live rate including overhead.',
        full_corpus_equivalent_hours=[dict(parameters=a['parameters'],hours=[
            train_rows/speed/3600*(a['parameters']/9967360) for speed in [3692.8,3200]])
            for a in result['architectures']])
    result['training_exposure'] = [dict(step=step,
        nominal_examples=(20000*64+(step-20000)*128),
        estimated_target_tokens=(20000*64+(step-20000)*128)*result['code']['mean_capped'],
        note='Nominal effective batch counts; partial batches and repeated sampled rows mean '
        'this is not a unique-example or exact-token count.') for step in [160000,180000]]
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--train-rows',type=int,default=11838003)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=estimate(args.pool,args.tokenizer,args.train_rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
