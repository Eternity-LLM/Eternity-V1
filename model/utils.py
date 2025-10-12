from hmac import new
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import rearrange, repeat
from kernels.quant import ActQuantFn, Fp8GemmFn, Fp8IndexFn
from kernels.ssm.ssd_combined import MambaChunkScanCombined

def act_quant(x:torch.Tensor):
    # Arguments:
    #     x (torch.Tensor): input tensor to be quantized
    # Returns:
    #     torch.Tensor: Output tensor after quantization
    return ActQuantFn.apply(x)

def weight_dequant(weight:torch.Tensor, scale:torch.Tensor, block_size:int = 128):
    shape = weight.shape
    assert weight.dim() == 2
    weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size).transpose(1, 2).contiguous().view(-1, block_size * block_size)
    weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype()).view(shape[0] // block_size, shape[1] // block_size, block_size, block_size).transpose(1, 2).contiguous().view(shape)
    return weight


def fp8_gemm(a:torch.Tensor, a_s:torch.Tensor, b:torch.Tensor, b_s:torch.Tensor):
    # Arguments:
    #     a (torch.Tensor): the first input matrix, must be contiguous
    #     a_s (torch.Tensor): the scaling factor of the first input matrix, must be contiguous
    #     b (torch.Tensor): the second input matrix, must be contiguous
    #     b_s (torch.Tensor): the scaling factor of the second input matrix, must be contiguous
    # Returns:
    #     torch.Tensor: the result of the matrix multiplication
    return Fp8GemmFn.apply(a, a_s, b, b_s)

def fp8_index(q:torch.Tensor, q_s:torch.Tensor, k:torch.Tensor, k_s:torch.Tensor):
    # FP8 index function for DeepSeek Sparse Attention (DSA)
    return Fp8IndexFn.apply(q, q_s, k, k_s)

# State Space Attention (SSA)
# This algorithm is based on SSD algorithm, however, I made a few changes.
# SSA is actually equivalent to linear attention $Y=L \circ \left (Q K^T \right ) \cdot V$,
# where $L$ is a lower triangular causal mask.
# For more details about the original SSD algorithm, please read the paper arXiv:2405.21060v1 (Dao and Gu, 2024).
def ssa(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        block_len:int = 64,
        initial_states:Optional[torch.Tensor] = None,
        return_final_states:bool = True
    ) -> Union[Tuple[torch.Tensor, None], Tuple[torch.Tensor, torch.Tensor]]:
    #Arguments:
    #    Q: (batch, length, n_heads, d_head)
    #    K: (batch, length, n_heads, d_head)
    #    V: (batch, length, n_heads, d_value)
    #Return:
    #    Y: (batch, length, n_heads, d_value)
    #    new_states: (batch, 1, n_heads, d_head, d_value)
    
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
    if V.shape[1] < block_len:
        block_len = 1
    if V.shape[1] % block_len != 0:
        len_1 = (V.shape[1] // block_len) * block_len
        _, final_states_1 = ssa(Q[:, :len_1], K[:, :len_1], V[:, :len_1], block_len = block_len, initial_states = initial_states)
        Y, final_states = ssa(Q[:, len_1:], K[:, len_1:], V[:, len_1:], block_len = 1, initial_states = final_states_1)
        return Y, final_states

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
    
    return Y, (new_states if return_final_states else None)

# State Space Duality (SSD)
# See the paper arXiv:2405.21060v1 (Dao and Gu, 2024) for more details.
def ssd(
        dt:torch.Tensor, 
        X:torch.Tensor, 
        A:torch.Tensor, 
        B:torch.Tensor, 
        C:torch.Tensor, 
        block_len:int = 64,
        initial_states:Optional[torch.Tensor] = None, 
        return_final_states:bool = True
    ) -> Union[Tuple[torch.Tensor, None], Tuple[torch.Tensor, torch.Tensor]]:
    
    return MambaChunkScanCombined.apply(X, dt, A, B, C, block_len, initial_states=initial_states, return_final_states=return_final_states)

def rotate_activation(x:torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16)
    hid_sz = x.shape[-1]
    return x * (hid_sz ** -0.5)

def qk_clip(attn_scores:torch.Tensor, max_val:int)->torch.Tensor:
    pass
