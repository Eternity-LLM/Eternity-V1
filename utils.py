import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat

def f(x:torch.Tensor):
    idx_1 = x>=0.0
    idx_2 = x<0.0
    x[idx_1] = (x[idx_1] + 1.0) ** 2
    x[idx_2] = 1.0 / ((x[idx_2] - 1.0) ** 2 )
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

# State Space Attention (SSA)
# This algorithm is based on SSD algorithm, however, I made a few changes.
# SSA is actually equivalent to linear attention $Y=L \circ \left (Q K^T \right ) \cdot V$,
# where $L$ is a lower triangular causal mask.
# For more details about the original SSD algorithm, please read the paper arXiv:2405.21060v1 (Dao and Gu, 2024).
def ssa(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, block_len:int = 64, initial_states:Optional[torch.Tensor] = None):
    #Arguments:
    #    Q: (batch, length, n_heads, d_head)
    #    K: (batch, length, n_heads, d_head)
    #    V: (batch, length, n_heads, d_value)
    #Return:
    #    Y: (batch, length, n_heads, d_value)
    
    # einsum notation:
    # b: batch_size
    # l: sequence length
    # h: number of heads
    # d: dimension of Q K head
    # n: dimension of V head
    # c: number of chunks
    # s: block length of query
    # t: block length of key/value

    assert Q.dtype == K.dtype == V.dtype
    assert V.shape[1] % block_len == 0, "sequence length must be divisible by block_len"

    # 1. Rearrange into chunks
    Q, K, V = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (Q, K, V)]

    # 2. Compute causal mask
    L = torch.tril(torch.ones(block_len, block_len, device=Q.device, dtype=Q.dtype))

    # 3. Compute the output for each intra-chunk (diagonal blocks)
    Y_diag = torch.einsum('bcshd,bcthd,st,bcthn->bcshn', Q, K, L, V)

    # 4. Compute the state for off-diagonal SSM recurrence
    states = torch.einsum('bcthd,bcthn->bchdn', K, V)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    states = torch.cumsum(states, dim=1)
    states, new_states = states[:, :-1], states[:, -1:]

    # 5. Compute off-diagonal SSM recurrence
    Y_off = torch.einsum('bcshd,bchdn->bcshn', Q, states)

    # 6. Compute the final output
    Y = rearrange(Y_diag + Y_off, "b c s h n -> b (c s) h n")
    return Y, new_states

# State Space Dual (SSD) algorithm    
# The following code is copied from https://github.com/state-spaces/mamba/
# For more information about state space dual (SSD) models, see arXiv:2405.21060v1 (Dao and Gu, 2024).

# Copyright (c) 2024, Albert Gu and Tri Dao.

# Minimal implementation of SSD.
# This is the same as Listing 1 from the paper.

def segsum(x):
    # More stable segment sum calculation.
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len:int = 64, initial_states=None):
    #Arguments:
    #    X: (batch, length, n_heads, d_head)
    #    A: (batch, length, n_heads)
    #    B: (batch, length, n_heads, d_state)
    #    C: (batch, length, n_heads, d_state)
    #Return:
    #    Y: (batch, length, n_heads, d_head)
    
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1:]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
