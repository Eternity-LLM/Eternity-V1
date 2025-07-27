import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

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

# state space attention (SSA)
class StateSpaceAttentionFunction(torch.autograd.Function):
    # State Space Attention Mechanism
    # Inputs:
    #  query: (batch_size, seq_len, num_heads, qk_head_dim)
    #  key:   (batch_size, seq_len, num_heads, qk_head_dim)
    #  value: (batch_size, seq_len, num_heads, v_head_dim)
    #  linear_mask: (batch_size, seq_len, seq_len)
    #  init_states: Optional (batch_size, num_chunks, seq_len, num_heads, qk_head_dim, v_head_dim) or None
    @staticmethod
    def forward(ctx, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, linear_mask:torch.Tensor, init_states:Optional[torch.Tensor], block_len:int=64):
        assert query.dtype == key.dtype == value.dtype == linear_mask.dtype
        assert value.shape[1] % block_len == 0 or value.shape[1] < block_len
        assert key.shape[1] == value.shape[1]

        if value.shape[1] < block_len:
            if init_states is None:
                init_states = torch.zeros((query.shape[0], 1, key.shape[1], query.shape[2], key.shape[3], value.shape[3]), dtype=query.dtype, device=query.device)
            init_states = rearrange(init_states, 'b c l ... -> b (c l) ...', l=block_len) # (batch_size, seq_len, num_heads, qk_head_dim, v_head_dim)
            KV = torch.einsum('blhk,blhv->blhkv', key, value)  # (batch_size, seq_len, num_heads, qk_head_dim, v_head_dim)
            states = init_states + KV
            

        