import torch
import torch.nn.functional as F

def f(x):
	idx_1 = x>=0.
    idx_2 = x<0.
    x[idx_1] = (x[idx_1] + 1.) ** 2
    x[idx_2] = 1. / ((x[idx_2] - 1.) ** 2 )
    return x

def f_sigmoid(x, s:float = 3.0):
    # "sigmoid" function with function f instead of exp
    return 1./(1.+f(-x*s))

def f_silu(x, s:float = 3.0):
    # "silu" function with function f instead of exp
    return x*f_sigmoid(x, s)

def f_softmax(x, s:float = 3.0, dim:int = -1):
    # "softmax" function with function f instead of exp
    x = f(x*s)
    s = torch.sum(x, dim = dim, keepdim = True)
    return x / s

def m(x):
    idx_1 = x>=0.
    idx_2 = x<0.
    x[idx_1] = x[idx_1] + 1.
    x[idx_2] = 1. / (1. - x[idx_2])
    return x

def g(x):
    idx_1 = x<1.
    idx_2 = x>=1.
    x[idx_1] = 1.0-torch.rsqrt(x)
    x[idx_2] = torch.sqrt(x)-1.0
    return x

def loss(y, y_pred):
    bsz, seq_len, dim = y.shape
    y = y.reshape(bsz * seq_len, dim)
    y_pred = y_pred.reshape(bsz * seq_len, dim)
    loss = torch.zeros_like(y)
    for i in range(dim):
        loss[:, i] = - y[:, i] * g(y_pred[:, i]) - (1. - y[:, i]) * g(1. - y_pred[:, i])
    loss = loss.reshape(bsz, seq_len, dim)
    loss = torch.sum(loss, dim=-1)
    return loss

