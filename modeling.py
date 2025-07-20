# Eternity-V1 modeling.py
# Edited by Haozhe Xu (14), Eternity-LLM Organization (since 2025)
# Based on DeepSeek-V3 ( url: https://github.com/deepseek-ai/deepseek-v3/ )
# Note that we use hybrid model (Attention, MLP, MoE, SSM)

from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from .kernel import act_quant, weight_dequant, fp8_gemm
import torch
import math
from . import utils as u


world_size = 1
rank = 0
use_deepseek = True   # If true, part of the parameters will be fine-tuned from DeepSeek-R1-0528, but model architecture is still Eternity-V1 (hybird model).
block_size = 128
gemm_impl = 'bf16'

@dataclass
class ModelArgs:
    max_batch_size:int = 16
    max_seq_len:int = 163840 #524288
    dtype:Literal['fp8', 'bf16'] = 'bf16'
    # embedding
    dim:int = 7168
    vocab_size:int = 129280
    emb_lora_rank:int = 256
    # attention
    q_lora_rank:int = 1536
    kv_lora_rank:int = 512
    qk_nope_head_dim:int = 128
    qk_rope_head_dim:int = 64
    v_head_dim:int = 128
    n_heads:int = 128
    mscale:float = 1.0
    # mlp
    mlp_dim:int = 18432
    # moe
    moe_dim:int = 2048
    n_shared:int = 1
    n_routed:int = 256
    n_routed_group:int = 8
    n_topk_group:int = 4
    n_experts_per_tok:int = 8
    n_dense_layers:int = 3
    # rope
    beta_fast:int = 32
    beta_slow:int = 1
    rope_theta:float = 10000.0
    rope_factor:float = 40.0
    original_seq_len:int = 4096


class ParallelEmbedding(nn.Module):
    # Embedding layer with parallelism support across distributed processes and low-rank adaptation.

    # Args:
    #     vocab_size (int) : vocabulary size.
    #     dim (int) : Embedding dimension.
    #     init_weight (optional, torch.tensor or nn.Parameter) : DeepSeek-R1-0528 pretrained params.
    #     lora_rank (int)

    def __init__(self, vocab_size:int, dim:int, init_weight = None, lora_rank:int = 256) -> None :
        super().__init__()
        self.vocabs = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.st_idx = rank * self.part_vocab_size
        self.en_idx = self.st_idx + self.part_vocab_size
        self.rank = lora_rank
        self.weight = nn.Parameter(init_weight[self.st_idx:(self.en_idx+1), :]) if init_weight else nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        if use_deepseek:
            self.weight.requires_grad = False
        
        if lora_rank:
            self.A = nn.Parameter(torch.empty(self.part_vocab_size, lora_rank))
            self.B = nn.Parameter(torch.empty(lora_rank, self.part_vocab_size))
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
        return None
    def forward(self, x:torch.Tensor):
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        if self.rank:
            y = F.embedding(x, self.weight + torch.matmul(self.A, self.B))
        else:
            y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

class Linear(nn.Module):
    # Custom linear layer
    defalt_dtype = torch.bfloat16
    
    def __init__(self, in_features:int, out_features:int, bias:bool = False, dtype = None) -> None :
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype = dtype or Linear.defalt_dtype))
        self.__bias = bias
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x:torch.Tensor):
        if self.weight.element_size() > 1:
            weighted_sum = F.linear(x, self.weight, None)
        elif gemm_impl == 'bf16':
            weighted_sum = F.linear(x, weight_dequant(self.weight, self.weight.scale))
        else:
            x, scale = act_quant(x, block_size)
            weighted_sum = fp8_gemm(x, scale, weight, weight.scale)
        if self.__bias:
            weighted_sum += self.bias
        return weighted_sum

class ColumnParallelLinear(Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = False, dtype = None) -> None :
        assert out_features % world_size == 0
        self.sz = out_features // world_size
        super().__init__(in_features, self.sz, bias = bias, dtype = dtype)
    def forward(self, x:torch.Tensor):
        return super().forward(x)

class RowParallelLinear(Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = False, dtype = None) -> None :
        assert in_features % world_size == 0
        self.sz = in_features // world_size
        super().__init__(self.sz, out_features, bias = bias,dtype = dtype)
        self.__bias_0 = bias
        self.__bias = False
    def forward(self, x:torch.Tensor):
        y = super().forward(x)
        if world_size > 1 :
            dist.all_reduce(y)
        if self.__bias_0 :
            y += self.bias
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x:torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

# RoPE
def precompute_freqs_cis(args:ModelArgs) -> torch.Tensor :
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    bf = args.beta_fast
    bs = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(n_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (n_rotations * 2 * math.pi)) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)
    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max += 0.001
        linear_func = (torch.arange(dim, dtype = torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(bf, bs, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

	t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(x:torch.Tensor, freqs_cis:torch.Tensor) -> torch.Tensor :
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    
