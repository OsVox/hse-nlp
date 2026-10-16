import pytest
from homeworks.hw2.length_cap_inference import affected_rows, PLAN


def test_limits_and_eos_choose_only_information_losing_rows():
    assert affected_rows([[3]*128,[3]*129,[3]],[[0,4,1],[0,4,1],[0,4,5]],128,1)==[1,2]
    assert affected_rows([[3]],[[0,1,0,0]],128,1)==[]
    with pytest.raises(ValueError):affected_rows([[3]],[],128,1)


def test_preregistered_regeneration_preserves_most_rows_and_is_bounded():
    assert len(PLAN['affected_rows'])==23
    assert PLAN['affected_rows']==sorted(set(PLAN['affected_rows']))
    assert PLAN['input_limit']==192 and PLAN['output_limit']==384
    assert not PLAN['training'] and not PLAN['gpu']
