"""Fit and screen lossless tokenizers on original course data using CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import time

import pyarrow.parquet as pq
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL = ['<pad>', '</s>', '<unk>']
PYTHON_SPLITS = r'\r?\n[ \t]*|[ \t]+|[A-Za-z_][A-Za-z_0-9]*|[0-9]+(?:\.[0-9]+)?|(?:\*\*|//|==|!=|<=|>=|<<|>>|:=|->)|[^\sA-Za-z_0-9]'
VARIANTS = {
    'balanced8192': dict(code_repeats=1, vocab_size=8192, python_boundaries=False),
    'code8192': dict(code_repeats=3, vocab_size=8192, python_boundaries=False),
    'python8192': dict(code_repeats=3, vocab_size=8192, python_boundaries=True),
    'code16384': dict(code_repeats=3, vocab_size=16384, python_boundaries=False),
}


def build_tokenizer(rows, *, code_repeats, vocab_size, python_boundaries):
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = (pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PYTHON_SPLITS), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        if python_boundaries else pre_tokenizers.ByteLevel(add_prefix_space=False))
    tok.decoder = decoders.ByteLevel()
    def texts():
        for row in rows:
            yield row['question']
            for _ in range(code_repeats):
                yield row['code']
    tok.train_from_iterator(texts(), trainer=trainers.BpeTrainer(vocab_size=vocab_size,
        min_frequency=2, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    tok.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>',1)])
    return tok


def metrics(tok, rows):
    tok.no_padding(); tok.no_truncation()
    result = {}
    began = time.monotonic()
    for column, limit in [('question',128),('code',192)]:
        lengths, numeric_pieces = [], []
        failures = unknowns = 0
        for start in range(0,len(rows),512):
            texts=[r[column] for r in rows[start:start+512]]
            for text, encoded in zip(texts,tok.encode_batch(texts)):
                lengths.append(len(encoded.ids))
                failures += tok.decode(encoded.ids,skip_special_tokens=True)!=text
                unknowns += encoded.ids.count(tok.token_to_id('<unk>'))
                for match in re.finditer(r'[0-9]+(?:\.[0-9]+)?',text):
                    numeric_pieces.append(sum(b>match.start() and a<match.end() for a,b in encoded.offsets))
        ordered=sorted(lengths)
        result[column]=dict(rows=len(rows),mean_tokens=statistics.mean(lengths),
            median_tokens=statistics.median(lengths),p99_tokens=ordered[int(.99*(len(ordered)-1))],
            over_limit=sum(x>limit for x in lengths),token_limit=limit,
            roundtrip_failures=failures,unknown_tokens=unknowns,
            numeric_spans=len(numeric_pieces),mean_pieces_per_numeric_span=statistics.mean(numeric_pieces) if numeric_pieces else None)
    result.update(vocab_size=tok.get_vocab_size(),evaluation_seconds=time.monotonic()-began)
    return result


def collect_rows(cache, *, max_shards=60, fit_per_shard=1024, eval_per_shard=256, heldout_rows=1024):
    heldout=set(pq.read_table(cache/'part-00000.parquet',columns=['question']).slice(0,heldout_rows)['question'].to_pylist())
    fit, candidates, provenance = [], [], []
    for shard in range(max_shards):
        path=cache/f'part-{shard:05d}.parquet'
        table=pq.read_table(path,columns=['question','code'])
        allowed=[i for i,q in enumerate(table['question'].to_pylist()) if q not in heldout]
        indices=random.Random(2026101601+shard).sample(allowed,min(len(allowed),fit_per_shard+eval_per_shard))
        fit_indices=indices[:fit_per_shard];eval_indices=indices[fit_per_shard:]
        fit.extend(table.take(fit_indices).to_pylist())
        candidates.extend((shard,i,row) for i,row in zip(eval_indices,table.take(eval_indices).to_pylist()))
        provenance.append(dict(shard=shard,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fit_row_indices=fit_indices))
    fit_questions={r['question'] for r in fit}
    evaluation=[];seen=set();evaluation_indices=[]
    for shard,i,row in candidates:
        if row['question'] in fit_questions or row['question'] in seen:
            continue
        seen.add(row['question']);evaluation.append(row);evaluation_indices.append([shard,i])
    assert not (seen & fit_questions) and not (seen & heldout) and not (fit_questions & heldout)
    return fit,evaluation,dict(shards=provenance,evaluation_indices=evaluation_indices,
        fit_rows=len(fit),evaluation_rows=len(evaluation),discarded_eval_candidates=len(candidates)-len(evaluation),
        disjoint_exact_question_texts=True,heldout_question_texts_excluded=True)


def audit(cache, original, output):
    output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    fit,evaluation,provenance=collect_rows(cache)
    (output/'sampling.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results={'original8192':metrics(Tokenizer.from_file(str(original/'tokenizer.json')),evaluation)}
    results['original8192']['tokenizer_sha256']=hashlib.sha256((original/'tokenizer.json').read_bytes()).hexdigest()
    for name,options in VARIANTS.items():
        t0=time.monotonic();tok=build_tokenizer(fit,**options)
        destination=output/name;destination.mkdir()
        tok.save(str(destination/'tokenizer.json'))
        for filename in ['tokenizer_config.json','special_tokens_map.json']:
            shutil.copyfile(original/filename,destination/filename)
        # Evaluate the serialized artifact used by any later training run.
        result=metrics(Tokenizer.from_file(str(destination/'tokenizer.json')),evaluation)
        result.update(options=options,fit_and_evaluation_seconds=time.monotonic()-t0,
            tokenizer_sha256=hashlib.sha256((destination/'tokenizer.json').read_bytes()).hexdigest())
        results[name]=result
        (output/'progress.json').write_text(json.dumps({'completed':list(results),'results':results},indent=2))
        print('TOKENIZER',name,json.dumps(result),flush=True)
    for name,r in results.items():
        r['code_token_change_vs_original_percent']=100*(r['code']['mean_tokens']/results['original8192']['code']['mean_tokens']-1)
        r['source_token_change_vs_original_percent']=100*(r['question']['mean_tokens']/results['original8192']['question']['mean_tokens']-1)
    result=dict(results=results,fit_rows=len(fit),evaluation_rows=len(evaluation),
        elapsed_seconds=time.monotonic()-began,
        original_overlap_note='Original tokenizer was previously trained on shard0; this is a compression diagnostic, not an unbiased generalization comparison against that tokenizer.',
        constraints='Only original course question/code strings. Code weighting repeats unchanged original strings. No pretrained tokenizer, target rewrite or new example. All new candidates use identical fit/evaluation question sets.',
        limitations='Shorter sequences or fewer number pieces do not prove better accuracy. New vocabulary IDs require fresh matched training; existing checkpoint token meanings must not be silently replaced.',
        research='https://github.com/huggingface/blog/blob/main/codeparrot.md')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('CPU-only tokenizer audit')
    audit(args.cache,args.original,args.output)
