import unittest
import torch
from homeworks.hw2.copyability_audit import count_group


class CopyabilityAuditTest(unittest.TestCase):
    def test_opportunity_is_distinct_from_pointer_recovery(self):
        # Two wrong reference tokens are available; the pointer recovers only one.
        mask=torch.tensor([[True,True,True,False]])
        result=count_group(mask,torch.tensor([[True,False,False,False]]),
            torch.tensor([[True,True,True,True]]),torch.tensor([[False,True,False,True]]))
        self.assertEqual(result,dict(tokens=3,correct=1,source_copyable=3,errors=2,
            copyable_errors=2,pointer_correct=1,pointer_correct_on_errors=1))


if __name__=='__main__':unittest.main()
