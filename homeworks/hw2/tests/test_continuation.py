"""Check the learning-rate phase without consuming accelerator time."""
import copy
import unittest
import torch
from homeworks.hw2.train import continuation_scheduler

class ContinuationTests(unittest.TestCase):
    def test_moments_survive_and_new_lr_replaces_old_base(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        opt=torch.optim.AdamW([p],lr=.001)
        p.grad=torch.tensor(2.);opt.step()
        moments=copy.deepcopy(opt.state[p])
        opt.param_groups[0]['initial_lr']=.001
        scheduler=continuation_scheduler(opt,learning_rate=.0005,start_step=20000,end_step=60000,warmup=500)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.0005/500)
        for key,value in moments.items():
            torch.testing.assert_close(opt.state[p][key],value)
        for _ in range(500):
            opt.step();scheduler.step()
        self.assertAlmostEqual(opt.param_groups[0]['lr'],.0005)
        self.assertAlmostEqual(scheduler.lr_lambdas[0](40000),.1)

    def test_restored_phase_matches_uninterrupted_rates(self):
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.001)
        scheduler=continuation_scheduler(opt,learning_rate=.0005,start_step=20000,end_step=60000,warmup=500)
        for _ in range(700):opt.step();scheduler.step()
        state=copy.deepcopy(scheduler.state_dict())
        restored=continuation_scheduler(opt,learning_rate=.0005,start_step=20000,end_step=60000,warmup=500)
        restored.load_state_dict(state)
        for group,lr in zip(opt.param_groups,restored.get_last_lr()):group['lr']=lr
        scheduler.step();restored.step()
        self.assertEqual(scheduler.get_last_lr(),restored.get_last_lr())

if __name__=='__main__':unittest.main()
