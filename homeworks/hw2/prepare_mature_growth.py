"""Prepare immutable paired parents for a bounded mature-model width test.

This CPU-only command never starts Colab, trains, generates public answers, or
marks the GPU experiment ready. The runtime controller is prepared separately.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from .checkpoint_io import archive_checkpoint, sha256


def plan_from_parent(config, step, model_hash):
    expected = dict(optimizer='adamw', seed=42, steps=184912, warmup=0,
        learning_rate=3e-5, schedule_start=92456, batch_size=128, accumulation=1,
        untruncated_training=True, group_window=32, input_pipeline='encoded',
        prefetch_batches=2, microbatch_token_budget=40960,
        microbatch_attention_budget=9961472)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Unexpected mature parent setting: '+key)
    model = config['model_config']
    for key, value in dict(d_model=512, d_ff=1536, num_heads=8, num_layers=6,
                          num_decoder_layers=4, vocab_size=8192, feed_forward_proj='gated-gelu').items():
        if model.get(key) != value:
            raise ValueError('Unexpected parent architecture: '+key)
    if step != 122962:
        raise ValueError('Use the selected held-out parent, not the public-selected checkpoint')
    return dict(name='mature-growth-300s-v1', parent_step=step,
        parent_model_sha256=model_hash, parent_selection='Best qualifying held-out checkpoint from completed selected-scale-v1',
        arms=['control42', 'grown66'], measured_training_seconds_per_arm=300,
        checkpoint_training_seconds=150, total_measured_training_seconds=600,
        per_arm_timeout_seconds=750, outer_timeout_seconds=1800,
        grown_d_ff=3072, growth_seed=42, growth_parameters=66074112,
        original_parameters=42481152, logical_batch=128, precision='bf16',
        reset_optimizer=False, restart_schedule=False, preserve_data_cursor=True,
        full_sequences=True, unchanged_original_targets=True,
        optimizer_migration='Copy old AdamW moments and steps; zero added coordinates, sharing the mature tensor step counter.',
        budget_comparison='Matched measured GPU training time; report steps, examples and complete elapsed time. Different model throughput means different prefix lengths of the same original stream.',
        selection='Compare final 300-second checkpoints on both fixed held-out subsets. Require strict gains in both and no increase in total prediction failures versus the paired control. Also report changes versus the untrained parent.',
        long_training_requires='A qualifying result and a subsequent independent paired confirmation; no automatic long run.',
        public_test_used_for_selection=False, gpu_launch_ready=False)


def prepare(parent, output, evidence, scale_run, probe_parquet, *, plan_fn=plan_from_parent):
    import pyarrow.parquet as pq
    import torch
    from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
    from .grow_ffn import grow_feed_forward, grow_adamw_optimizer

    if torch.cuda.is_available() or output.exists() or evidence.exists():
        raise ValueError('Require CPU and fresh output/evidence paths')
    torch.set_num_threads(2)
    began = time.monotonic()
    manifest = json.loads((parent/'checkpoint.complete.json').read_text())
    for name, digest in manifest['files'].items():
        assert Path(name).name == name and sha256(parent/name) == digest
    config = json.loads((parent/'run_config.json').read_text())
    step = json.loads((parent/'step.json').read_text())['step']
    plan = plan_fn(config, step, manifest['files']['model.safetensors'])
    completion = json.loads((scale_run/'execution.complete.json').read_text())
    outcome = json.loads((scale_run/'execution.json').read_text())
    progress = json.loads((scale_run/'progress.json').read_text())
    assert completion['returncode'] == 0 and outcome['completed']
    assert outcome['best_model_sha256'] == progress['state']['best_model_sha256'] == plan['parent_model_sha256']
    ready = json.loads((scale_run/'ready.json').read_text())
    assert sha256(probe_parquet) == ready['dataset']['parquet_files'][probe_parquet.name]
    state = torch.load(parent/'training_state.pt', map_location='cpu', weights_only=False)
    assert state['step'] == step and state['scheduler']['last_epoch'] == step-config['schedule_start']
    model = T5ForConditionalGeneration.from_pretrained(parent, local_files_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    optimizer.load_state_dict(state['optimizer'])
    grown = grow_feed_forward(model, plan['grown_d_ff'], seed=plan['growth_seed'])
    migrated = grow_adamw_optimizer(model, grown, optimizer)
    assert sum(p.numel() for p in model.parameters()) == plan['original_parameters']
    assert sum(p.numel() for p in grown.parameters()) == plan['growth_parameters']
    assert all(s['step'].item() == step for s in migrated.state.values())
    # Probe original dataset rows; no gradient update or generated target.
    examples = next(pq.ParquetFile(probe_parquet).iter_batches(batch_size=4)).to_pylist()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(parent, local_files_only=True)
    inputs = tokenizer([r['question'] for r in examples], padding=True,
        return_tensors='pt', return_token_type_ids=False)
    labels = tokenizer([r['code'] for r in examples], padding=True,
        return_tensors='pt', return_token_type_ids=False)['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    model.eval(); grown.eval()
    with torch.inference_mode():
        a = model(**inputs, labels=labels).logits
        b = grown(**inputs, labels=labels).logits
        difference = float((a-b).abs().max())
        assert torch.equal(a, b), 'CPU function-preservation probe failed'
    output.mkdir(parents=True)
    records = {}
    with tempfile.TemporaryDirectory(prefix='hw2-mature-parents-') as temp:
        for arm in plan['arms']:
            raw = Path(temp)/arm; raw.mkdir()
            if arm == 'control42':
                for name in manifest['files']:
                    shutil.copyfile(parent/name, raw/name)
            else:
                grown.save_pretrained(raw, safe_serialization=True)
                for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                             'coverage.json', 'step.json']:
                    shutil.copyfile(parent/name, raw/name)
                new_config = deepcopy(config)
                new_config['model_config'] = grown.config.to_dict()
                new_config['growth_preparation'] = dict(parent_model_sha256=plan['parent_model_sha256'],
                    parent_training_state_sha256=manifest['files']['training_state.pt'],
                    original_d_ff=1536, grown_d_ff=3072, growth_seed=plan['growth_seed'],
                    optimizer_policy=plan['optimizer_migration'])
                (raw/'run_config.json').write_text(json.dumps(new_config, indent=2)+'\n')
                torch.save(dict(state, optimizer=migrated.state_dict()), raw/'training_state.pt')
            marker = archive_checkpoint(raw, output/arm)
            if arm == 'control42':
                assert marker['files'] == manifest['files']
            for name in ['coverage.json', 'step.json', 'tokenizer.json']:
                assert marker['files'][name] == manifest['files'][name]
            records[arm] = dict(model_sha256=marker['files']['model.safetensors'],
                training_state_sha256=marker['files']['training_state.pt'],
                checkpoint_manifest_sha256=sha256(output/arm/'checkpoint.complete.json'))
    archive = output/'paired-parents.tar'
    with tarfile.open(archive, 'w') as tar:
        for arm in plan['arms']:
            tar.add(output/arm, arcname=arm)
    for name, digest in manifest['files'].items():
        assert sha256(parent/name) == digest
    evidence.mkdir(parents=True)
    (evidence/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    proof = dict(status='paired_parents_prepared', parent_step=step,
        parent_model_sha256=plan['parent_model_sha256'], parent_unchanged=True,
        arm_checkpoints=records, archive_sha256=sha256(archive), archive_bytes=archive.stat().st_size,
        original_row_probe_count=len(examples), max_absolute_logit_difference=difference,
        probe_parquet_sha256=sha256(probe_parquet), cpu_only=True, training_updates=0,
        public_predictions_generated=False, gpu_launch_ready=False,
        script_sha256=sha256(Path(__file__)), growth_helper_sha256=sha256(Path(__file__).with_name('grow_ffn.py')),
        torch_version=torch.__version__, elapsed_seconds=time.monotonic()-began,
        limitation='Four-row CPU numerical check and immutable parent preparation; no quality result or CUDA/BF16 equivalence claim.')
    (evidence/'parents-verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['parent', 'output', 'evidence', 'scale-run', 'probe-parquet']:
        p.add_argument('--'+name, type=Path, required=True)
    prepare(**vars(p.parse_args()))
