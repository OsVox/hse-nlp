"""One bounded AdamW78 arm, compared with the verified AdamW42 reference."""
import argparse
from copy import deepcopy
import json

from . import learning_curve
from .checkpoint_io import sha256

NAME = 'adamw-size-curve-v1'
PREPARED = learning_curve.ROOT/'prepared'/NAME
RUN = learning_curve.ROOT/'runs'/NAME
PLAN = deepcopy(learning_curve.PLAN)
PLAN.update(
    arms={'adamw78':dict(d_model=704,d_ff=2112,num_heads=11,optimizer='adamw')},
    outer_timeout_seconds=2100,
    hypothesis='Complete the missing AdamW size comparison: width704 versus width512 with the optimizer held fixed.',
    training_budget='One fresh AdamW78 arm: 600 measured training seconds, checkpoints at 300 and 600 seconds. Reuse the verified AdamW42 600-second reference to save ten GPU minutes.',
    selection='Require strict gains on both fixed held-out subsets without added total execution failures. A winner requires a subsequent paired second-seed AdamW42/AdamW78 run before long training.',
    reference='Archived AdamW42 step6018 from matched-curve-v1, same original tokenizer, fresh exclusion index, runtime versions, GPU model and 600-second budget. Validate stage identities before comparison.',
    limitations='The reference was measured in a previous runtime. Equal nominal GPU and software do not guarantee identical host load or throughput. Report examples processed and timing separately; confirm any winner in a paired fresh run. Early results do not prove the best long-run size.',
    second_seed_required_for_promotion=True,
    reuse_reference=True,
    prerequisite='Finish and preserve the active CPU averaging screen before any GPU allocation.')
LAUNCH = (learning_curve.LAUNCH.replace(learning_curve.NAME,NAME)
    .replace('homeworks.hw2.learning_curve','homeworks.hw2.adamw_size_curve')
    .replace('timeout=3900','timeout=2100')
    .replace('MATCHED_CURVE_STARTED','ADAMW_SIZE_STARTED')
    .replace('Hard bound3900seconds; three arms,600training seconds each, two checkpoints per arm.',
             'Hard bound2100seconds; one AdamW78 arm,600training seconds, two checkpoints.'))

STAGE_IDENTITY = ('questions_sha256','evaluation_examples_sha256','evaluator_sha256',
    'offset','rows','usable_references','beams','device','precision','batch_size',
    'max_input_tokens','max_new_tokens','handwritten_repairs')


def select(results, reference):
    if set(results) != {'adamw78'}:
        raise ValueError('Expected only the predeclared AdamW78 arm')
    candidate = results['adamw78']
    control = reference['results']
    for stage in ('pilot','confirmation'):
        for key in STAGE_IDENTITY:
            if candidate[stage][key] != control[stage][key]:
                raise ValueError(f'Incomparable {stage} measurement: {key}')
    comparison = learning_curve.comparison(candidate,control)
    return dict(results={'adamw42_archived':control,'adamw78':candidate},
        comparison=comparison,provisional_winner='adamw78' if comparison['passes'] else 'adamw42',
        second_seed_required=comparison['passes'],reference_reused=True,
        reference_model_sha256=reference['model_sha256'],public_test_used_for_selection=False,
        long_run_started=False,full_pass_completed=False)


def run():
    import torch
    ready=json.loads((PREPARED/'ready.json').read_text())
    assert ready['cpu_verified'] and ready['prerequisite_cpu_averaging_completed']
    assert ready['plan'] == PLAN
    reference_path=PREPARED/'reference.json'
    assert sha256(reference_path) == ready['reference_sha256']
    reference=json.loads(reference_path.read_text())
    assert torch.cuda.get_device_name() == reference['gpu']
    assert reference['budget']['optimizer'] == 'adamw'
    assert 600 <= reference['budget']['training_seconds'] < 610
    assert ready['fresh_preparation']['index_manifest_sha256'] == reference['index_manifest_sha256']

    def checked_selection(results):
        budget=json.loads((RUN/'adamw78/compute-budget.json').read_text())
        checkpoint=RUN/'adamw78'/f"step-{budget['final_step']:06d}"
        config=json.loads((checkpoint/'run_config.json').read_text())
        assert config['optimizer']=='adamw' and config['d_model']==704
        assert config['initial_model_sha256']==reference['expected_78m_initial_sha256']
        return select(results,reference)

    learning_curve.run(prepared=PREPARED,run_dir=RUN,plan=PLAN,selection_fn=checked_selection)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true',required=True)
    p.parse_args();run()
