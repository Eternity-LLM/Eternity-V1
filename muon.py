import torch
from torch import nn
from torch.optim.optimizer import ParamsT

def newtonschulz5(G, steps:int=5, eps:float=1e-7):
    assert G.ndim == 2, f'Number of dimensions ({G.ndim}) is not available.'
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params:ParamsT, lr:float=1e-3, momentum:float=0.95, weight_decay:float=0.1) -> None:
        defaults = dict(
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay
        )
        super().__init__(params=params, defaults=defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_m_buffer'] = torch.zeros_like(p.data)
                state['scale'] = float(max(p.size())) ** 0.5
    
    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                M_t = momentum * state['momentum_m_buffer'] + grad
                state['momentum_m_buffer'] = M_t
                if M_t.ndim != 2:
                    M_t = M_t.reshape(M_t.size(0), -1)
                O_t = newtonschulz5(M_t) * state['scale'] * 0.2
                p.data.sub_(lr*(O_t.reshape(*p.size()) + weight_decay * p.data))
        return loss
