"""Native PyTorch Muon on hidden matrices, AdamW on embeddings and vectors.

Reference: https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html
No optimizer implementation is copied; require torch >= 2.9 with native Muon.
"""
import math
import torch


def partition_parameters(model):
    # T5 has 2D relative-position embeddings as well as tied input/output weights.
    # Dimensionality alone would incorrectly send those embeddings to Muon.
    excluded = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding)
                for p in m.parameters(recurse=False)}
    if hasattr(model, 'get_output_embeddings'):
        output = model.get_output_embeddings()
        if output is not None:
            excluded.update(id(p) for p in output.parameters())
    matrices, other = [], []
    names = {'muon': [], 'adamw': []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        use_muon = p.ndim == 2 and id(p) not in excluded
        (matrices if use_muon else other).append(p)
        names['muon' if use_muon else 'adamw'].append(name)
    assert len({id(p) for p in matrices + other}) == len(matrices) + len(other)
    return matrices, other, names


class MuonAdamW(torch.optim.Optimizer):
    """One scheduler/checkpoint interface over two disjoint native optimizers."""
    def __init__(self, model, *, lr, muon_lr, weight_decay=.01):
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('Native Muon requires torch >= 2.9; prepare dependencies on CPU first')
        if not all(math.isfinite(v) and v > 0 for v in [lr, muon_lr]):
            raise ValueError('Learning rates must be finite and positive')
        matrices, other, self.parameter_names = partition_parameters(model)
        if not matrices or not other:
            raise ValueError('Expected both hidden matrices and non-Muon parameters')
        super().__init__(matrices + other, {})
        self.muon = torch.optim.Muon(matrices, lr=muon_lr, weight_decay=weight_decay,
            momentum=.95, nesterov=True, ns_steps=5, adjust_lr_fn='match_rms_adamw')
        self.adamw = torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay)
        self.muon.param_groups[0]['lr_scale'] = muon_lr / lr
        self.adamw.param_groups[0]['lr_scale'] = 1.
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self):
        return dict(format='native_muon_adamw_v1', parameter_names=self.parameter_names,
                    muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, state_dict):
        if state_dict.get('format') != 'native_muon_adamw_v1':
            raise ValueError('Cannot reinterpret AdamW moments as Muon momentum')
        if state_dict['parameter_names'] != self.parameter_names:
            raise ValueError('Optimizer parameter names/order changed')
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def make_optimizer(model, name, *, learning_rate, muon_learning_rate=None):
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    if name == 'muon':
        return MuonAdamW(model, lr=learning_rate,
                         muon_lr=muon_learning_rate or learning_rate)
    raise ValueError(name)
