"""Bounded complete-sequence accuracy curve from our own best checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from . import broad_screen, coverage_pilot
from .async_scoring import ScoringJobs
from .checkpoint_io import sha256
from .short_pilots import ROOT, CACHE, copy_verified

NAME='complete-curve-4k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
INDEX=Path('/content/complete-curve-inputs/coverage-index')
STEPS=[73728,74752,75776,76800]
PLAN=deepcopy(coverage_pilot.PLAN)
PLAN.update(final_step=76800,save_every=1024,eval_every=1024,
    training_timeout_seconds=720,outer_timeout_seconds=900,
    added_example_presentations_per_arm=483328,checkpoint_steps=STEPS,
    arms={'complete':dict(numeric_token_weight=2.,mixed_rows_per_shard=None,
        coverage_index=str(INDEX),untruncated_training=True,
        microbatch_token_budget=40960,microbatch_attention_budget=9961472)},
    hypothesis='Measure a bounded accuracy curve after retaining complete original sequences and exact consumed-data accounting.',
    matched='One continuation from the best own 42M weights and AdamW moments. Seed42, batch128, BF16, numeric weight2, warmup50 and peak rate3e-5. No parameter scaling or new targets. This is not a causal comparison against capped training.',
    selection='Report all four held-out checkpoints against the parent. Public scores are recorded separately. No automatic extension or full-pass claim; inspect the trajectory before any longer continuation.',
    limitations='At most 483328 newly tracked original rows, about4.08% of the corpus. Old exposure is unknown. Held-out sets are reused. Validation and strict greedy caps remain128/192 for comparability.',
    next_stages=['Verify every complete-token coverage receipt.','Export and bot-score all four retained checkpoints.','Review held-out trajectory before extending.'])
LAUNCH=(coverage_pilot.LAUNCH.replace(coverage_pilot.NAME,NAME)
    .replace('homeworks.hw2.coverage_pilot','homeworks.hw2.complete_curve')
    .replace('COVERAGE_PIPELINE_STARTED','COMPLETE_CURVE_STARTED')
    .replace('Hard bound 900 seconds; two matched 1024-update arms only.',
             'Hard bound 900 seconds; one 3776-update continuation, four retained checkpoints.'))


def restore_index(run_dir):
    ready=json.loads((PREPARED/'ready.json').read_text())
    archive=INDEX.parent/'index.tar.gz';archive.parent.mkdir(exist_ok=False)
    copy_verified(Path(ready['coverage_index_archive_path']),archive)
    assert sha256(archive)==ready['coverage_index_sha256']
    with tarfile.open(archive) as tar:tar.extractall(archive.parent,filter='data')
    assert sha256(INDEX/'manifest.json')==ready['coverage_manifest_sha256']
    (run_dir/'index-verification.json').write_text(json.dumps(dict(
        archive_sha256=ready['coverage_index_sha256'],
        manifest_sha256=ready['coverage_manifest_sha256']),indent=2))


def verify_coverage(record,step):
    r=record['report'];expected=(step-PLAN['parent_step'])*PLAN['batch_size']
    assert record['optimizer_step']==step and r['start_step']==PLAN['parent_step']
    assert r['presentations']==r['unique_rows_since_tracking_started']==expected
    assert r['repeated_presentations_since_tracking_started']==r['completed_full_passes']==0
    assert r['source_length'] is None and r['target_length'] is None
    assert r['untruncated_tokens']==record['state']['untruncated_tokens']
    assert all(r['untruncated_tokens'][key]>=expected for key in ['question','code'])
    return r


def select(final_results):
    parent=json.loads((PREPARED/'ready.json').read_text())['parent_heldout']
    results={str(step):{} for step in STEPS};receipts={};jobs=ScoringJobs()
    previous_tokens={'question':0,'code':0}
    try:
        for step in STEPS:
            name=str(step);checkpoint=RUN/'complete'/f'step-{step:06d}'
            record=json.loads((checkpoint/'coverage.json').read_text())
            receipts[name]=verify_coverage(record,step)
            for key,value in receipts[name]['untruncated_tokens'].items():
                assert value>previous_tokens[key]
            previous_tokens=receipts[name]['untruncated_tokens']
            provenance=json.loads((checkpoint/'predictions.provenance.json').read_text())
            assert provenance['num_beams']==1 and provenance['syntax_retry_beams']==0
            assert (checkpoint/'results.complete.json').exists()
            if step==STEPS[-1]:
                results[name]=final_results['complete'];continue
            for stage,offset,rows in [('pilot',256,256),('confirmation',512,512)]:
                output=checkpoint/(stage+'.json')
                command=[sys.executable,'-m','homeworks.hw2.numeric_eval',
                    '--model-dir',str(checkpoint),'--cache-dir',str(CACHE),
                    '--output',str(output),'--offset',str(offset),'--rows',str(rows),'--beams','1']
                with output.with_suffix('.log').open('w') as log:
                    subprocess.run(command+['--generate-only'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                jobs.start(name,stage,command+['--score-only'],output)
                jobs.collect(results)
        jobs.collect(results,wait=True)
    finally:
        jobs.close()
        (RUN/'curve-scoring-execution.json').write_text(json.dumps(jobs.completed,indent=2))
    trajectory=[]
    for step in STEPS:
        scores=results[str(step)]
        trajectory.append(dict(step=step,results=scores,coverage=receipts[str(step)],
            parent_gains={s:scores[s]['numeric_correct']-parent[s]['numeric_correct'] for s in parent},
            failure_delta=sum(scores[s]['prediction_failures']-parent[s]['prediction_failures'] for s in parent)))
    return dict(trajectory=trajectory,parent_heldout=parent,all_checkpoints_evaluated=True,
        own_scratch_checkpoint=True,public_test_used_for_selection=False,
        automatic_extension=False,completed_training_full_pass=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True);p.parse_args()
    broad_screen.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=select,
                     runtime_preflight_fn=restore_index)
