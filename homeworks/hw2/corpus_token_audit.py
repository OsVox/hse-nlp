"""Count untruncated tokens for every eligible original training row on CPU.

Lengths are stored in eligible-index order. No code is executed or rewritten,
and no model/runtime is allocated. Per-shard artifacts permit verified resume.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


CAPS = [(64, 192), (96, 192), (128, 192), (192, 256), (192, 384), (256, 512)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for b in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def histogram_summary(histogram):
    pairs = sorted((int(n), int(c)) for n, c in histogram.items())
    total = sum(c for _, c in pairs)
    if not total:
        return dict(rows=0, total_tokens=0, mean_tokens=None, quantiles={})
    lengths, counts = np.asarray(pairs, dtype=np.int64).T
    cumulative = np.cumsum(counts)
    # Match numpy.quantile(method='higher') without expanding 12M rows.
    quantiles = {str(q): int(lengths[np.searchsorted(cumulative,
        int(np.ceil(q * (total - 1))) + 1)]) for q in [.5, .9, .95, .99, .999, 1.]}
    tokens = sum(n*c for n,c in pairs)
    return dict(rows=total, total_tokens=tokens, mean_tokens=tokens/total,
                quantiles=quantiles)


def cap_counts(source, target):
    result = {}
    for s,t in CAPS:
        result[f'{s}:{t}'] = dict(
            source_cut=int((source>s).sum()), target_cut=int((target>t).sum()),
            either_cut=int(((source>s)|(target>t)).sum()),
            both_cut=int(((source>s)&(target>t)).sum()),
            source_tokens_removed=int(np.maximum(source.astype(np.int64)-s,0).sum()),
            target_tokens_removed=int(np.maximum(target.astype(np.int64)-t,0).sum()))
    return result


def count_texts(tokenizer, texts, batch_size=1024):
    lengths = np.empty(len(texts), dtype=np.uint32)
    specials = {'unknown_token_occurrences': 0, 'literal_pad_occurrences': 0,
                'rows_with_internal_eos': 0}
    eos, unk, pad = [tokenizer.token_to_id(s) for s in ['</s>', '<unk>', '<pad>']]
    assert eos is not None
    for start in range(0,len(texts),batch_size):
        batch = texts[start:start+batch_size]
        if any(not isinstance(text,str) for text in batch):
            raise ValueError('Null/non-string original text; audit must not skip rows')
        encoded = tokenizer.encode_batch(batch)
        for offset,encoding in enumerate(encoded):
            ids=encoding.ids
            if not ids or ids[-1]!=eos:
                raise ValueError('Unexpected end-token convention')
            lengths[start+offset]=len(ids)
            specials['unknown_token_occurrences'] += ids.count(unk)
            specials['literal_pad_occurrences'] += ids.count(pad)
            specials['rows_with_internal_eos'] += eos in ids[:-1]
    return lengths, specials


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def audit(cache, index, tokenizer_path, output):
    began=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((index/'manifest.json').read_text())
    if manifest['version']!=1 or not manifest['shards']:
        raise ValueError('Unsupported/empty eligible index')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_padding();tokenizer.no_truncation()
    identity=dict(index_manifest_sha256=sha256(index/'manifest.json'),
        tokenizer_sha256=sha256(tokenizer_path),audit_source_sha256=sha256(__file__),
        versions={p:version(p) for p in ['tokenizers','numpy','pyarrow']})
    records=[]; hist={'question':Counter(),'code':Counter()}
    totals={f'{s}:{t}':Counter() for s,t in CAPS}
    for expected, record in enumerate(manifest['shards']):
        if record['shard']!=expected or Path(record['filename']).name!=record['filename']:
            raise ValueError('Invalid shard identity')
        path=cache/record['filename'];ids_path=index/f'{expected:05d}.npy'
        if sha256(path)!=record['parquet_sha256'] or sha256(ids_path)!=record['index_sha256']:
            raise ValueError('Input checksum mismatch')
        ids=np.load(ids_path,allow_pickle=False)
        if (ids.dtype!=np.int64 or len(ids)!=record['eligible'] or
                (len(ids) and (ids[0]<0 or ids[-1]>=record['rows'])) or
                np.any(ids[1:]<=ids[:-1])):
            raise ValueError('Invalid eligible row indices')
        result_path=output/f'shard-{expected:05d}.json';length_path=output/f'shard-{expected:05d}.npz'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if (result['identity']!=identity or result['record']!=record or
                    sha256(length_path)!=result['length_file_sha256']):
                raise ValueError('Changed audit input or artifact; use a new output directory')
        else:
            started=time.monotonic();table=pq.read_table(path,columns=['question','code'])
            if table.num_rows!=record['rows']:
                raise ValueError('Parquet row count mismatch')
            table=table.take(ids)
            q,qs=count_texts(tokenizer,table['question'].to_pylist())
            c,cs=count_texts(tokenizer,table['code'].to_pylist())
            np.savez_compressed(length_path,question=q,code=c)
            result=dict(identity=identity,record=record,length_file_sha256=sha256(length_path),
                length_order='Ascending eligible original row IDs from index file',
                question_specials=qs,code_specials=cs,caps=cap_counts(q,c),
                histogram={name:{str(int(n)):int(count) for n,count in zip(*np.unique(a,return_counts=True))}
                           for name,a in [('question',q),('code',c)]},
                elapsed_seconds=time.monotonic()-started)
            atomic_json(result_path,result)
        for name in hist:hist[name].update({int(k):v for k,v in result['histogram'][name].items()})
        for name in totals:totals[name].update(result['caps'][name])
        records.append(result)
        atomic_json(output/'progress.json',dict(completed_shards=len(records),total_shards=len(manifest['shards']),
            eligible_rows=sum(r['record']['eligible'] for r in records),elapsed_seconds=time.monotonic()-began))
        print(json.dumps({'shard':expected,'rows':record['eligible'],'elapsed_seconds':result['elapsed_seconds'],
            'current_caps':result['caps']['128:192']}),flush=True)
    rows=sum(r['record']['eligible'] for r in records)
    for name in hist:assert sum(hist[name].values())==rows
    result=dict(identity=identity,eligible_rows=rows,shards=len(records),gpu_used=False,
        training_changed=False,token_count_includes_eos=True,truncation=False,padding=False,
        results={name:histogram_summary(h) for name,h in hist.items()},
        caps={k:{**dict(v),'either_cut_percent':100*v['either_cut']/rows} for k,v in totals.items()},
        special_tokens={name:{key:sum(r[name+'_specials'][key] for r in records)
            for key in records[0][name+'_specials']} for name in hist},
        elapsed_seconds_this_invocation=time.monotonic()-began,
        shard_processing_seconds=sum(r['elapsed_seconds'] for r in records),
        scope='Every eligible original row in the verified index; excludes held-out question copies. No test answers read.',
        limitations=['Row length counts do not measure how many of these rows a checkpoint has trained on.',
            'A longer context may reduce truncation but its speed and quality must be measured.',
            'Removing comments or changing labels is not performed by this audit.'])
    atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=Path('homeworks/hw2/tokenizer/tokenizer.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit(args.cache,args.index,args.tokenizer,args.output),indent=2))
