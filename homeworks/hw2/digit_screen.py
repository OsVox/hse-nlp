"""Small adaptation test: original BPE versus individual digit boundaries."""
import argparse
from copy import deepcopy
import random
import statistics
import time

from . import broad_screen
from .checkpoint_io import sha256
from .short_pilots import ROOT

NAME = 'digit-screen-1k-v1'
PREPARED = ROOT/'prepared'/NAME
RUN = ROOT/'runs'/NAME
PLAN = deepcopy(broad_screen.PLAN)
PLAN.update(final_step=71024, save_every=71024, eval_every=71024,
            added_example_presentations_per_arm=131072, training_updates_per_arm=1024,
            arms={'control': {}, 'digits': {'individual_digits': True}},
            outer_timeout_seconds=900,
            hypothesis='Individual digit boundaries may make copying and composing numeric literals easier. Original vocabulary, IDs and text are unchanged.',
            limitations='Longer sequences change length grouping and truncation; the exact example pool matches at the complete-window boundary. Adaptation from BPE is not a from-scratch tokenizer comparison. Held-out questions have been reused.',
            selection='Positive gain in both held-out subsets, no increase in total execution failures; then repeat with another seed before combination or extension. No automatic long run.',
            research='https://arxiv.org/abs/2402.14903; motivates testing segmentation, does not establish benefit for Python generation')
LAUNCH = (broad_screen.LAUNCH.replace('broad-screen-1k-v1', NAME)
          .replace('homeworks.hw2.broad_screen', 'homeworks.hw2.digit_screen')
          .replace('timeout=2400', 'timeout=900')
          .replace('SHORT_SCREEN_STARTED', 'DIGIT_SCREEN_STARTED')
          .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
                   'Hard bound 900 seconds; two matched 1024-update arms only.'))


def audit_tokenization(parent, cache):
    import pyarrow.parquet as pq
    from .data import training_shard, validation_examples, SHARDS, VALIDATION_ROWS
    from .digit_tokenization import load_training_tokenizer
    started = time.monotonic()
    old = load_training_tokenizer(parent)
    new = load_training_tokenizer(parent, individual_digits=True)
    assert old.get_vocab() == new.get_vocab() and old.all_special_ids == new.all_special_ids
    heldout = {r['question'] for r in validation_examples(cache)}
    lengths = {mode: {field: [] for field in ('question','code')} for mode in ('original','digits')}
    rows = 0
    for shard in range(SHARDS):
        table = pq.read_table(training_shard(shard, cache), columns=['question','code'])
        first = VALIDATION_ROWS if shard == 0 else 0
        indices = random.Random(20261016 + shard).sample(range(first, len(table)), 2048)
        sample = [r for r in table.take(indices).to_pylist() if r['question'] not in heldout]
        rows += len(sample)
        for field in ('question','code'):
            texts = [r[field] for r in sample]
            for name, tokenizer in [('original',old),('digits',new)]:
                ids = tokenizer(texts, truncation=False)['input_ids']
                assert all(x[-1] == tokenizer.eos_token_id for x in ids)
                decoded = tokenizer.batch_decode([x[:-1] for x in ids], skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
                assert decoded == texts, (shard,field,name,'roundtrip differs')
                lengths[name][field].extend(map(len,ids))
    stats = {mode: {field: dict(mean=statistics.mean(values), median=statistics.median(values),
                              truncated=sum(n > (128 if field=='question' else 192) for n in values))
                    for field,values in fields.items()} for mode,fields in lengths.items()}
    saved = parent.parent/'digits-tokenizer-preflight'
    new.save_pretrained(saved)
    restored = load_training_tokenizer(saved)
    assert restored.get_vocab() == old.get_vocab()
    assert restored(['12345.67','def solve():\n    return 1000\n']) == new(['12345.67','def solve():\n    return 1000\n'])
    return dict(rows=rows,shards=SHARDS,rows_per_shard_before_exclusion=2048,
                exact_original_text_roundtrip=True,unchanged_vocabulary_and_ids=True,
                checkpoint_reload_verified=True,stats=stats,
                original_tokenizer_sha256=sha256(parent/'tokenizer.json'),
                digit_tokenizer_sha256=sha256(saved/'tokenizer.json'),
                elapsed_seconds=time.monotonic()-started,training_started=False)


def select(results):
    result = broad_screen.select_candidates(results)
    # Both diagnostic subsets must improve before spending on confirmation.
    result['shortlist'] = [r['arm'] for r in result['ranking']
                           if r['eligible'] and min(r['split_gains']) > 0]
    result['training_updates_per_arm'] = 1024
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true')
    if parser.parse_args().gpu:
        broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select)
    else:
        broad_screen.prepare(prepared=PREPARED,run_dir=RUN,plan=PLAN,launcher=LAUNCH,
                             preflight_fn=audit_tokenization)
