import copy
import pytest
from homeworks.hw2.complete_curve import PLAN, STEPS, LAUNCH, verify_coverage
from homeworks.hw2.broad_screen import training_command


def test_curve_has_bounded_complete_checkpoint_schedule():
    saved=[s for s in range(PLAN['parent_step']+1,PLAN['final_step']+1)
           if s%PLAN['save_every']==0 or s==PLAN['final_step']]
    assert saved==STEPS
    command=training_command(PLAN['arms']['complete'])
    assert '--untruncated-training' in command and '--mixed-rows-per-shard' not in command
    assert '--predict-at-checkpoint' in command and '--reset-optimizer' not in command
    compile(LAUNCH,'launch.py','exec')


@pytest.mark.parametrize('fault',['step','cut','duplicates','tokens'])
def test_coverage_claim_rejects_inconsistent_receipts(fault):
    n=(STEPS[0]-PLAN['parent_step'])*128
    tokens={'question':n*40,'code':n*70}
    record=dict(optimizer_step=STEPS[0],state={'untruncated_tokens':copy.deepcopy(tokens)},
        report=dict(start_step=PLAN['parent_step'],presentations=n,
            unique_rows_since_tracking_started=n,repeated_presentations_since_tracking_started=0,
            completed_full_passes=0,source_length=None,target_length=None,untruncated_tokens=tokens))
    verify_coverage(record,STEPS[0])
    if fault=='step':record['optimizer_step']+=1
    if fault=='cut':record['report']['target_length']=192
    if fault=='duplicates':record['report']['unique_rows_since_tracking_started']-=1
    if fault=='tokens':record['state']['untruncated_tokens']['code']-=1
    with pytest.raises(AssertionError):verify_coverage(record,STEPS[0])
