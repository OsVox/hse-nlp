import unittest

from homeworks.hw2.grouped_training import end_step, select_quality, PILOT_EXAMPLES, TOTAL_EXAMPLES


class BudgetTests(unittest.TestCase):
    def test_changed_batch_preserves_full_example_budget(self):
        # Global update numbers are not comparable after a batch change.
        # These cases catch accidentally allocating 20k updates at every size.
        for batch, final_step in [(64,34000),(128,20000),(256,13000),(512,9500)]:
            pilot_end=end_step(6000,batch,PILOT_EXAMPLES)
            result=end_step(pilot_end,batch,TOTAL_EXAMPLES-6000*128-PILOT_EXAMPLES)
            self.assertEqual(result,final_step)
            self.assertEqual(6000*128+(result-6000)*batch,TOTAL_EXAMPLES)

    def test_quality_gate_rejects_loss_or_execution_regression(self):
        def record(correct,loss):
            return dict(numeric=dict(numeric_correct=correct,usable_references=245),
                        validation=dict(token_loss=loss))
        control=record(50,.8)
        self.assertFalse(select_quality(control,record(49,.7)))
        self.assertFalse(select_quality(control,record(51,.82)))
        self.assertTrue(select_quality(control,record(51,.8)))


if __name__=='__main__':unittest.main()
