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
def chunk_linear_mask(linear_mask:torch.Tensor, block_len:int=64):
    # not completed yet
    pass

class StateSpaceAttentionFunction(torch.autograd.Function):
    # State Space Attention Mechanism
    # Note that this mechanism is equivalent to the linear attention mechanism: $ \bm{Y} = \bm{L} \circ (\bm{Q} \cdot \bm{K}^T) \cdot \bm{V} $ , 
    # where $\circ$ is the element-wise product, $\bm{L}$ is the linear mask, $\bm{Q}$ is the query, $\bm{K}$ is the key, and $\bm{V}$ is the value.
    
    # Inputs:
    #  query: (batch_size, seq_len, num_heads, qk_head_dim)
    #  key:   (batch_size, seq_len, num_heads, qk_head_dim)
    #  value: (batch_size, seq_len, num_heads, v_head_dim)
    #  linear_mask: (batch_size, seq_len, seq_len)
    #  init_states: Optional (batch_size, num_chunks, seq_len, num_heads, qk_head_dim, v_head_dim) or None
    @staticmethod
    def forward(ctx, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, linear_mask:torch.Tensor, init_states:Optional[torch.Tensor]=None, block_len:int=64):
        # einsum notation:
        # b = batch size
        # c = num_chunks
        # l = seq_len (key and value)
        # h = num_heads
        # k = qk_head_dim (key and query)
        # v = v_head_dim (value)
        # q = query len (query)
        
        assert query.dtype == key.dtype == value.dtype == linear_mask.dtype
        assert value.shape[1] % block_len == 0 or value.shape[1] < block_len
        assert key.shape[1] == value.shape[1]

        if value.shape[1] < block_len:
            # still developing, not completed yet
            pass
        
        ctx.save_for_backward(query, key, value, linear_mask)

        # 1. Rearrange the inputs into chunks
        Q, K, V = [rearrange(x, 'b (c l) ... -> b c l ...', l = block_len) for x in (query, key, value)]
        L_diag, L_off = chunk_linear_mask(linear_mask, block_len=block_len)

        # 2. Compute the dot-product of key and value
        KV = torch.einsum('bclhk,bclhv->bclhkv', K, V)

        # 3. Compute diagonal chunks
        Y_diag = torch.einsum('bcqhk,bclhkv,bcql->bcqhv', Q, KV, L_diag)

        # 4. Compute states for off-diagonal SSM recurrence
        if init_states is None:
            init_states = torch.zeros_like(KV[:,:1])
        states = torch.cat([init_states, KV], dim=1)
        states = torch.cumsum(states, dim=1)
        states, new_states = states[:,:-1], states[:,-1:]

        # 5. Compute off-diagonal SSM recurrence
        Y_off = torch.einsum('bcqhk,bclhkv->bcqhv', Q, states)

        # 6. Compute final output
        Y = rearrange(Y_diag + Y_off, 'b c l h v -> b (c l) h v')

        return Y, new_states
    
    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        query, key, value, linear_mask = ctx.saved_tensors
        dQ = torch.einsum('bqhv,bql,blhk,blhv->bqhk', grad_output, linear_mask, key, value)
        dK = torch.einsum('bqhv,bql,bqhk,blhv->blhk', grad_output, linear_mask, query, value)
        dV = torch.einsum('bqhv,bql,bqhk,blhk->blhv', grad_output, linear_mask, query, key)
        return dQ, dK, dV, None, None, None