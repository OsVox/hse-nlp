"""Short scratch-model comparison of original and code-fitted tokenizers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from .async_scoring import ScoringJobs
from .broad_screen import LAUNCH as BASE_LAUNCH, select_candidates
from .checkpoint_io import drive_preflight, sha256
from .short_pilots import ROOT, CACHE, restore_data

NAME='representation-pilot-2k-v1'
PREPARED=ROOT/'prepared'/NAME
RUN=ROOT/'runs'/NAME
PLAN=dict(steps=2048,batch_size=128,accumulation=1,seed=42,warmup=200,
    learning_rate=.0003,save_every=2048,eval_every=2048,
    precision='bf16',group_window=32,mixed_rows_per_shard=4096,
    input_pipeline='encoded',prefetch_batches=2,prediction_batch_size=256,
    d_model=512,d_ff=1536,encoder_layers=6,decoder_layers=4,num_heads=8,
    initialization_vocab_size=16384,
    arms={'control':'original8192','code8192':'code8192','code16384':'code16384','comments':'original8192'},
    training_updates_per_arm=2048,example_presentations_per_arm=262144,
    training_timeout_seconds=400,outer_timeout_seconds=1600,
    hypothesis='Independently test code-fitted vocabulary, larger vocabulary, and removal of comments/empty lines for early executed-answer learning. Comments arm retains the original tokenizer.',
    matched='Identical random Transformer body and embedding row prefix via common 16k initialization; reset dropout seed after shrinking. Same original example stream and complete grouping-window budget. Tokenization changes sorting within windows.',
    selection='Require positive held-out numeric gain on both subsets and no added total execution failures versus current-tokenizer scratch control. Confirm survivors with a second seed before extension. No automatic long run.',
    limitations='Early scratch learning does not establish the eventual best tokenizer. Vocabulary size changes parameter count and output cost. Token losses across tokenizers are not directly comparable. Current-tokenizer control differs from new8k in fitting sample and code weight.',
    decoding='Strict greedy, num_beams=1, no repairs',
    provenance='All random weights and course examples only. Comments arm replaces each existing target once with an identical-AST version, without new examples or alternate solutions. No pretrained vocabulary or external training data.')
LAUNCH=(BASE_LAUNCH.replace('broad-screen-1k-v1',NAME)
    .replace('homeworks.hw2.broad_screen','homeworks.hw2.tokenizer_pilot')
    .replace('timeout=2400','timeout=1600')
    .replace('SHORT_SCREEN_STARTED','TOKENIZER_PILOT_STARTED')
    .replace('Hard bound 2400 seconds; eight matched 1k arms only.',
             'Hard bound 1600 seconds; four scratch 2048-update arms only.'))


def select(results):
    result=select_candidates(results)
    for row in result['ranking']:
        row['eligible'] = row['eligible'] and min(row['split_gains']) > 0
    result['shortlist']=[r['arm'] for r in result['ranking'] if r['eligible']]
    result['training_updates_per_arm']=2048
    result['training_from_scratch']=True
    return result


def run(*, prepared=PREPARED, run_dir=RUN, plan=PLAN, selection_fn=select):
    PREPARED, RUN, PLAN, select = prepared, run_dir, plan, selection_fn
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.cuda.get_device_properties(0).total_memory>=40*1024**3
    drive_preflight(ROOT)
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['plan']==PLAN and not RUN.exists()
    assert Path('source_commit.txt').read_text().strip()==ready['source_commit']
    for name, files in ready['tokenizers'].items():
        for filename,digest in files.items():
            assert sha256(Path('pilot-tokenizers')/name/filename)==digest
    RUN.mkdir(parents=True);(RUN/'runtime-manifest.json').write_text(json.dumps(ready,indent=2))
    started=time.monotonic();status={'completed':False,'gpu':torch.cuda.get_device_name()}
    scoring=ScoringJobs();results={};initial_hashes={}
    with (RUN/'gpu-utilization.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--loop=1'],stdout=telemetry)
        try:
            restore_data(ready['dataset'])
            option_names=['steps','batch_size','accumulation','seed','warmup','learning_rate','save_every','eval_every','precision','group_window','mixed_rows_per_shard','input_pipeline','prefetch_batches','prediction_batch_size','d_model','d_ff','encoder_layers','decoder_layers','num_heads','initialization_vocab_size']
            for arm_name,tokenizer_name in PLAN['arms'].items():
                arm=RUN/arm_name;arm.mkdir()
                options={k:PLAN[k] for k in option_names}
                options.update(tokenizer_dir=Path('pilot-tokenizers')/tokenizer_name,
                    cache_dir=CACHE,output_dir=Path('/content/tokenizer-pilot-training')/arm_name,drive_root=arm)
                command=[sys.executable,'-u','-m','homeworks.hw2.train']
                for key,value in options.items():command.extend(['--'+key.replace('_','-'),str(value)])
                command.append('--predict-at-checkpoint')
                if arm_name == 'comments':command.append('--strip-comments')
                (arm/'launch.json').write_text(json.dumps({'command':command},indent=2))
                with (arm/'training.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=PLAN['training_timeout_seconds'])
                checkpoint=arm/'step-002048'
                assert (checkpoint/'results.complete.json').exists()
                initial_hashes[arm_name]=json.loads((checkpoint/'run_config.json').read_text())['initial_nonembedding_sha256']
                assert initial_hashes[arm_name] and len(set(initial_hashes.values()))==1
                provenance=json.loads((checkpoint/'predictions.provenance.json').read_text())
                assert provenance['num_beams']==1 and provenance['syntax_retry_beams']==0
                results[arm_name]={}
                for stage,offset,rows in [('pilot',256,256),('confirmation',512,512)]:
                    output=arm/(stage+'.json')
                    command=[sys.executable,'-m','homeworks.hw2.numeric_eval','--model-dir',str(checkpoint),'--cache-dir',str(CACHE),'--output',str(output),'--offset',str(offset),'--rows',str(rows),'--beams','1']
                    with output.with_suffix('.log').open('w') as log:
                        subprocess.run(command+['--generate-only'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                    scoring.start(arm_name,stage,command+['--score-only'],output)
                scoring.collect(results)
                (RUN/'progress.json').write_text(json.dumps({'results':results,'initial_body_hashes':initial_hashes},indent=2))
            scoring.collect(results,wait=True)
            (RUN/'selection.json').write_text(json.dumps(select(results),indent=2))
            (RUN/'initialization.json').write_text(json.dumps({'body_hashes':initial_hashes,'all_equal':len(set(initial_hashes.values()))==1},indent=2))
            status['completed']=True
        finally:
            scoring.close();monitor.terminate();monitor.wait(timeout=10)
            (RUN/'scoring-execution.json').write_text(json.dumps(scoring.completed,indent=2))
            status['elapsed_seconds']=time.monotonic()-started
            (RUN/'execution.json').write_text(json.dumps(status,indent=2))
            with zipfile.ZipFile(RUN/'exports.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(RUN.rglob('*')):
                    if path.is_file() and path.suffix in {'.json','.jsonl','.csv','.log'}:archive.write(path,path.relative_to(RUN))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true',required=True)
    parser.parse_args();run()
