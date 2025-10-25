# Eternity-V1 modeling.py
# Edited by Haozhe Xu (14), Eternity-LLM Organization
# Based on DeepSeek-V3 (url https://github.com/deepseek-ai/deepseek-v3/)
# Note that we use hybrid model (Attention, MLP, MoE, SSM)

from ast import Tuple
from einops import rearrange
from sympy import Union
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
from typing import Optional, Literal
from utils import act_quant, qk_clip, weight_dequant, fp8_gemm, fp8_index, rotate_activation
import torch
import math
import utils as u
from muon import Muon
import sys


world_size = 1
rank = 0
block_size = 128
gemm_impl:Literal['bf16', 'fp8'] = 'bf16'
attn_impl:Literal['naive', 'absorb'] = 'naive'

@dataclass
class ModelArgs:
    # Model arguments and hyperparameters for Eternity-V1
    model_type = 'eternity_v1'
    def __init__(
        self,
        max_batch_size: int = 16,
        max_seq_len: int = 163840,
        dtype: Literal['fp8', 'bf16'] = 'bf16',
        scale_fmt: Optional[str] = None,


        # embedding
        dim: int = 7168,
        vocab_size: int = 129280,
        emb_lora_rank: int = 256,


        # sequence transformations
        # ssa (state space attention) and
        # csa (clipped sparse attention) and
        # lsm (lightning state space model)
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        n_heads: int = 128,
        mscale: float = 1.0,
        block_len: int = 64,
        A_init_range: Tuple[float, float] = (1.0, 16.0),
        

        # feed-forward networks
        # mlp (multi-layer perceptron)
        mlp_dim: int = 18432,
        mlp_conv_kernel_size:int = 16,
        # moe (mixture of experts)
        moe_dim: int = 2048,
        n_shared: int = 1,
        n_routed: int = 256,
        n_routed_group: int = 8,
        n_topk_group: int = 4,
        n_experts_per_tok: int = 8,
        n_dense_layers: int = 3,
        routed_scale: float = 2.5,
        shared_conv_kernel_size: int = 32,


        # rope (rotary position embedding)
        beta_fast: int = 32,
        beta_slow: int = 1,
        rope_theta: float = 10000.0,
        rope_factor: float = 40.0,
        original_seq_len: int = 4096,


        # indexer
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,


        # block
        n_csa_layers: int = 5,
        pure_attn_ratio: float = 0.08,
        n_blocks: int = 61,
        dropout_rate: float = 0.2
    ) -> None:
        
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.scale_fmt = scale_fmt


        # embedding
        self.dim = dim
        self.vocab_size = vocab_size
        self.emb_lora_rank = emb_lora_rank


        # sequence transformations
        # ssa (state space attention) and
        # csa (clipped sparse attention) and
        # lsm (lightning state space model)
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_heads = n_heads
        self.mscale = mscale
        self.block_len = block_len
        self.A_init_range = A_init_range
        

        # feed-forward networks
        # mlp (multi-layer perceptron)
        self.mlp_dim = mlp_dim
        self.mlp_conv_kernel_size = mlp_conv_kernel_size
        # moe (mixture of experts)
        self.moe_dim = moe_dim
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.n_routed_group = n_routed_group
        self.n_topk_group = n_topk_group
        self.n_experts_per_tok = n_experts_per_tok
        self.n_dense_layers = n_dense_layers
        self.routed_scale = routed_scale
        self.shared_conv_kernel_size = shared_conv_kernel_size


        # rope (rotary position embedding)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.original_seq_len = original_seq_len


        # indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk


        # block
        self.n_csa_layers = n_csa_layers
        self.pure_attn_ratio = pure_attn_ratio
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate

class ParallelEmbedding(nn.Module):
    # Embedding layer with parallelism support across distributed processes and low-rank adaptation.

    # Args:
    #     vocab_size (int) : vocabulary size.
    #     dim (int) : Embedding dimension.
    #     init_weight (optional, torch.tensor or nn.Parameter) : DeepSeek-R1-0528 pretrained params.
    #     lora_rank (int)

    # Outputs:
    #     Embedded tensor of shape (batch_size, seq_len, dim).

    def __init__(self, vocab_size:int, dim:int, init_weight = None, lora_rank:int = 256) -> None :
        super().__init__()
        self.vocabs = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.st_idx = rank * self.part_vocab_size
        self.en_idx = self.st_idx + self.part_vocab_size
        self.rank = lora_rank
        self.weight = nn.Parameter(init_weight[self.st_idx:(self.en_idx+1), :]) if init_weight is not None else nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

        if lora_rank:
            self.A = nn.Parameter(torch.empty(self.part_vocab_size, lora_rank))
            self.B = nn.Parameter(torch.empty(lora_rank, self.part_vocab_size))
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
        return None
    def forward(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor]=None):
        # Forward pass for parallel embedding layer
        # Args:
        #     x (torch.Tensor) : Input tensor containing token indices
        #     padding_mask (optional, torch.Tensor) : Mask for padding tokens with 1s for padding tokens and 0s for non-padding tokens
        # Returns:
        #     y (torch.Tensor) : Embedded representations
        # Raises:
        #     ValueError: If `world_size` is not defined
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
        if padding_mask is not None:
            y[padding_mask] = 0
        return y

def linear(x: torch.Tensor, weight: torch.Tensor, scale_fmt: Optional[str] = None) -> torch.Tensor:
    # Applies a linear transformation to the incoming data: $y=xA^T$
    # This function supports specialized implementations based on quantization
    # and tensor formats
    # Args:
    #     x (torch.Tensor) : The input tensor
    #     weight (torch.Tensor) : The weight tensor which may be quantized and requires
    #                             dequantization for certain cases
    #     scale_fmt (Optional[str]): The format of the scaling factor, default is None
    # Returns:
    #     torch.Tensor: The output tensor after applying the linear transformation

    if weight.dtype != torch.float8_e4m3fn:
        x = x.to(weight.dtype)
        return F.linear(x, weight)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        return fp8_gemm(x, scale, weight, weight.scale)

class Linear(nn.Module):
    dtype = torch.bfloat16
    scale_fmt:Optional[str] = None

    def __init__(self, in_features: int, out_features: int, dtype = None) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        self.register_parameter("bias", None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.use_scale)

class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, dtype)

class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x:torch.Tensor):
        return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps).type_as(x)

class LayerNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x:torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)

# RoPE (rotary position embedding)
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
    # print(x.size())
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # print(x.size())
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class Indexer(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank

        self.wq_b = Linear(self.dim, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.get_default_dtype())
        self.softmax_scale = self.head_dim ** (-0.5)
        self.scale_fmt = args.scale_fmt

        self.register_buffer('k_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer('k_scale_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)
    
    def forward(self, x:torch.Tensor, qr:torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        dist.broadcast(topk_indices_, src=0)
        assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices

class CSA(nn.Module):
    # Clipped Sparse Attention (CSA) Layer

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
        self.dequant_wkv_b = None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        # Args:
        #     x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        #     start_pos (int): Starting position in the sequence for caching.
        #     freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
        #     mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        # Returns:
        #     torch.Tensor: Output tensor with the same shape as the input.

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.wq_a(x)
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:    # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            scores = torch.einsum("bshd,bthd->bsht", q.float(), k.float()) * self.softmax_scale

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            index_mask += mask
            scores += index_mask.unsqueeze(2)

            # qk-clip
            # see Kimi-K2: Open Agentic Intelligence (arXiv:2507.20534)
            if self.training:
                scores = qk_clip(scores)

            scores = scores.softmax(dim=-1, dtype=torch.float32)
            x = torch.einsum("bsht,bthd->bshd", scores.type_as(x), v)
        else:                   # MHA decode
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_nope.float(), self.kv_cache[:bsz, :end_pos].float()) +
                      torch.einsum("bshr,btr->bsht", q_pe.float(), self.pe_cache[:bsz, :end_pos].float())) * self.softmax_scale

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            # qk-clip
            # see Kimi-K2: Open Agentic Intelligence (arXiv:2507.20534)
            if self.training:
                scores = qk_clip(scores)

            scores = scores.softmax(dim=-1, dtype=torch.float32)
            x = torch.einsum("bsht,btc->bshc", scores.type_as(x), self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

class SSA(nn.Module):
    # State Space Attention (SSA)
    # Note that this mechanism is equivalent to linear attention  mechanism: $ \bm{Y} = \bm{L} \circ (\bm{Q} \cdot \bm{K}^T) \cdot \bm{V} $ , 
    # where $\circ$ is the element-wise product, $\bm{L}$ is the linear mask, $\bm{Q}$ is the query, $\bm{K}$ is the key, and $\bm{V}$ is the value.
    # I made several changes to the state space dual (SSD) model. For more details of the original SSD model, please read the paper arXiv:2405.21060v1 (Dao and Gu, 2024).
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.block_len = args.block_len
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.scale = self.qk_head_dim ** (-0.5)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.csa_mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        
        self.register_buffer('kv_states', torch.zeros(args.max_batch_size, 1, self.n_local_heads, self.qk_head_dim, self.v_head_dim), persistent=False)
        
        self.q_norm = RMSNorm(self.q_lora_rank if self.q_lora_rank > 0 else self.dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]=None):
        bsz, seqlen, _ = x.size()
        
        if self.q_lora_rank == 0:
            q = self.wq(self.q_norm(x))
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, -1)
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = torch.cat((q_nope, q_pe), dim=-1)
        k = torch.cat((k, k_pe.expand(-1, -1, self.n_local_heads, -1)), dim=-1)

        q, k = torch.elu(q) + 1.0, torch.elu(k) + 1.0

        output, states = u.ssa(q.float(), k.float(), v.float(), initial_states=self.kv_states, block_len=self.block_len, return_final_states=not self.training)
        if not self.training:
            self.kv_states[:bsz] = states
        output = output * self.scale
        output = self.wo(output.view(bsz, seqlen, -1))
        return output
    
class LSM(nn.Module):
    # Lightning State Space Model (LSM)
    # Based on state space duality.
    # For more information, see the paper arXiv:2405.21060v1 (Dao and Gu, 2024).

    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        self.dim = args.dim
        self.block_len = args.block_len
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.c_lora_rank = args.q_lora_rank
        self.bx_lora_rank = args.kv_lora_rank
        self.cb_head_dim = args.qk_nope_head_dim
        self.cb_discretization_head_dim = args.qk_rope_head_dim
        self.x_head_dim = args.v_head_dim
        self.cb_proj_head_dim = self.cb_head_dim + self.cb_discretization_head_dim

        if self.c_lora_rank == 0:
            self.wc = ColumnParallelLinear(self.dim, self.n_heads * self.cb_head_dim)
        else:
            self.wc_a = Linear(self.dim, self.c_lora_rank)
            self.wc_b = ColumnParallelLinear(self.c_lora_rank, self.n_heads * self.cb_head_dim)
        self.wbx_a = Linear(self.dim, self.bx_lora_rank + self.cb_discretization_head_dim)
        self.wbx_b = ColumnParallelLinear(self.bx_lora_rank, self.n_heads * (self.cb_head_dim + self.x_head_dim))

        self.register_buffer('ssm_states', torch.zeros(args.max_batch_size, 1, self.n_local_heads, self.cb_head_dim, self.x_head_dim), persistent=False)

        # A parameter
        A_init_range = args.A_init_range
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=Linear.dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
    
    def forward(self, x:torch.Tensor):
        bsz, seqlen, _ = x.size()

        A = - torch.exp(self.A_log)

        if self.c_lora_rank == 0:
            c = self.wc(x)
        else:
            c = self.wc_b(self.wc_a(x))
        c = c.view(bsz, seqlen, self.n_local_heads, self.cb_proj_head_dim)
        c, dt_c = torch.split(c, [self.cb_head_dim, self.cb_discretization_head_dim], dim=-1)

        bx = self.wbx_a(x)
        bx, dt_b = torch.split(bx, [self.bx_lora_rank, self.cb_discretization_head_dim], dim=-1)
        bx = self.wbx_b(bx)

        bx = bx.view(bsz, seqlen, self.n_local_heads, self.cb_head_dim + self.x_head_dim)
        b, x = torch.split(bx, [self.cb_head_dim, self.x_head_dim], dim=-1)

        dt = dt_c @ dt_b

        outputs, states = u.ssd(dt, x, A, b, c, block_len=self.block_len, initial_states=self.ssm_states, return_final_states=not self.training)
        if not self.training:
            self.ssm_states[:bsz] = states
        return outputs

class ParallelSeperableConv1d(nn.Module):
    # still developing ...
    pass

class MLP(nn.Module):
    # Multi-Layer Perceptron (MLP) with seperable convolutional (opt.)
    def __init__(self, dim:int, mlp_dim:int, use_conv:bool = False, conv_kernel_size:int = 32, max_batch_size:int=16) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, mlp_dim)
        self.w2 = RowParallelLinear(mlp_dim, dim)
        self.w3 = ColumnParallelLinear(dim, mlp_dim)
        self.conv = None
        if use_conv:
            self.conv = ParallelSeperableConv1d(dim, mlp_dim, kernel_size=conv_kernel_size, max_batch_size=max_batch_size)
    
    def forward(self, x:torch.Tensor):
        h = self.w3(x)
        if self.conv is not None:
            h += self.conv(x)
        h = F.silu(self.w1(x)) * h
        return self.w2(h)

class Gate(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_experts_per_tok
        self.n_groups = args.n_routed_group
        self.n_topk_group = args.n_topk_group
        self.routed_scale = args.routed_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed))
    def forward(self, x:torch.Tensor):
        scores = linear(x, self.weight)
        scores = F.sigmoid(scores)
        
        original_scores = scores
        scores += self.bias
        if self.n_groups > 1 :
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.topk(2, dim = -1)[0].sum(dim = -1)
            indices = group_scores.topk(self.n_topk_group, dim = -1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim = -1)[1]
        weights = original_scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.routed_scale
        return weights.type_as(x), indices

class Expert(nn.Module):
    def __init__(self, dim:int, moe_dim:int):
        super().__init__()
        self.w1 = Linear(dim, moe_dim)
        self.w2 = Linear(moe_dim, dim)
        self.w3 = Linear(dim, moe_dim)
    def forward(self, x:torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    # Mixture of Experts (MoE)
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        assert args.n_routed % world_size == 0, f"Number of routed experts must be divisible by world size (world_size={world_size})"
        self.n_routed = args.n_routed
        self.n_local_routed = self.n_routed // world_size
        self.n_experts_per_tok = args.n_experts_per_tok
        self.start_idx = rank * self.n_local_routed
        self.end_idx = self.start_idx + self.n_local_routed
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_dim) if self.start_idx <= i < self.end_idx else None
                                        for i in range(self.n_routed)])
        self.shared = MLP(args.dim, args.n_shared * args.moe_dim, use_conv=True, conv_kernel_size = args.shared_conv_kernel_size, max_batch_size=args.max_batch_size) if args.n_shared > 0 else None
    
    def forward(self, x:torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()
        for i in range(self.start_idx, self.end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if world_size > 1:
            dist.all_reduce(y)
        if self.shared is not None:
            y = y + self.shared(x.view(shape)).view(y.shape)
        return y.view(shape)

class Block(nn.Module):
    # Eternity-V1 StateFormer Block
    def __init__(self, args:ModelArgs, layer_idx:int) -> None:
        super().__init__()
        if layer_idx < args.n_csa_layers:
            self.seq_transformation = CSA(args)
        else:
            if (layer_idx - args.n_csa_layers + 1) % (1 // args.pure_attn_ratio) == 0:
                self.seq_transformation = SSA(args)
            else:
                self.seq_transformation = LSM(args)
        self.ffn = MLP(
        	args.dim, args.mlp_dim, max_batch_size=args.max_batch_size, use_conv=True, conv_kernel_size=args.mlp_conv_kernel_size
        ) if layer_idx < args.n_dense_layers else MoE(args)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)
        self.dr = nn.Dropout(p=args.dropout_rate)
    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]=None):
        # Arguments:
        #     x: (batch_size, seq_len, dim) tensor of embedded tokens.
        #     start_pos: starting position for the sequence (default: 0).
        #     freqs_cis: precomputed freqs_cis of rotary position embedding (RoPE).
        #     mask: optional attention mask for sparse attention layers (default: None).
        h = self.dr(self.seq_transformation(self.norm1(x), start_pos, freqs_cis, mask))
        x = x + self.dr(self.ffn(self.norm2(h)))
        return x

class StateFormer(nn.Module):
    # Eternity-V1 StateFormer Model
    def __init__(self, args:ModelArgs) -> None :
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == 'fp8' else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.emb = ParallelEmbedding(args.vocab_size, args.dim, init_weight=None, lora_rank=args.emb_lora_rank)
        self.layers = torch.nn.ModuleList()
        self.n_blocks = args.n_blocks
        self.n_csa_layers = args.n_csa_layers
        self.ratio = args.pure_attn_ratio
        for idx in range(args.n_blocks):
            self.layers.append(Block(args, idx))
        self.norm = RMSNorm(args.dim)
        self.final = ColumnParallelLinear(args.dim, args.vocab_size, dtype = torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    
    # If you are using this model for inference, then add this line of code:
    # @torch.inference_mode()
    def forward(self, tokens:torch.Tensor, start_pos:int=0, attn_mask:Optional[torch.Tensor]=None, padding_mask:Optional[torch.Tensor]=None) -> None :
        # Eternity-V1 forward pass
        # Arguments:
        #   tokens: (batch_size, seq_len) tensor of tokens.
        #   start_pos: starting position for the sequence (default: 0).
        #   attn_mask: optional attention mask for sparse attention layers (default: None).
        #   padding_mask: optional padding mask for the input tokens, with 1s for padding tokens and 0s for non-padding tokens (default: None).
        # Outputs:
        #   logits: (batch_size, seq_len, vocab_size) tensor of logits for each token in the sequence.
        
        seqlen = tokens.size(1)
        bsz = tokens.size(0)
        h = self.emb(tokens, padding_mask)
        
        freqs_cis = self.freqs_cis[start_pos:(start_pos+seqlen)]
        n = 0

        with torch.no_grad:
            if attn_mask is None and seqlen >1:
                attn_mask = torch.full((bsz, seqlen, seqlen), float('-inf'), device=tokens.device, requires_grad=False).tril_(1)
            if padding_mask is not None:
                __p = padding_mask.unsqueeze(-1).expand(-1, -1, seqlen)
                attn_mask[__p] = float('-inf')
                attn_mask = attn_mask.transpose(-1, -2)
                attn_mask[__p] = float('-inf')
                attn_mask = attn_mask.transpose(-1, -2)
        
        for layer in self.layers:
            if n < self.n_csa_layers:
                h = layer(h, start_pos, freqs_cis, attn_mask)
            else:
                h = layer(h, start_pos, freqs_cis)
            n += 1
            
        h = self.norm(h)
        logits = self.final(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim = -1)
        return logits



# Model cold start decorator
# Usage:
# @cold_start(model)
# def your_function(...):
#     ...
# or, you can use it as a context manager:
# with cold_start(model):
#     ...

class __EternityV1ColdStartDecorator:
    model = None

    def __init__(self, func):
        self.func = func
    
    def set_model(self, model:StateFormer):
        self.model = model

    def __set_requires_grad(self, r:bool):
        try:
            model = self.model
            assert model is not None
        except:
            raise RuntimeError('Model not found.')
        # Set embedding layer
        model.emb.weights.requires_grad = False
        for layer in model.layers:
            seqt = layer.seq_transformation
            ffn = layer.ffn
            # Set attn layers
            if type(seqt).__name__ != 'LSM':
                for p in seqt.parameters():
                    p.requires_grad = r
            # Set ffn layers
            if type(ffn).__name__ == 'MLP':
                ffn.w1.weight.requires_grad = r
                ffn.w3.weight.requires_grad = r
            else:
                ffn.shared.w1.weight.requires_grad = r
                ffn.shared.w3.weight.requires_grad = r
                for expert in ffn.experts:
                    expert.w1.weight.requires_grad = r
                    expert.w2.weight.requires_grad = r
                    expert.w3.weight.requires_grad = r

    def __call__(self, *args, **kwargs):
        self.__set_requires_grad(False)
        return_val = self.func(*args, **kwargs)
        self.__set_requires_grad(True)
        return return_val
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def cold_start(model:StateFormer):
    decorator = __EternityV1ColdStartDecorator
    decorator.model = model
    return decorator

if '--test' in sys.argv or '-t' in sys.argv or '/t' in sys.argv:
    # Train the model in a small scale with random tokens
    # in order to find out the bugs in the program
    print(f'Testing modeling.py ({__name__}).')
    args = ModelArgs(
        max_batch_size = 2,
        max_seq_len = 256,
        dtype = 'bf16',
        dim = 512,
        vocab_size = 5,
        emb_lora_rank = 0,
        q_lora_rank = 32,
        kv_lora_rank = 16,
        qk_nope_head_dim = 64,
        qk_rope_head_dim = 16,
        v_head_dim = 64,
        n_heads = 8,
        mscale = 1.0,
        ssa_block_len = 64,
        csa_q_lora_rank = 64,
        csa_kv_lora_rank = 32,
        csa_mscale = 1.0,
        csa_max_attn_score = 100.0,
        gate_dim = 128,
        ssm_lora_rank = 16,
        mlp_dim = 512,
        moe_dim = 32,
        n_shared = 1,
        n_routed = 8,
        n_routed_group = 4,
        n_topk_group = 2,
        n_experts_per_tok = 2,
        n_dense_layers = 1,
        shared_conv_kernel_size = 3,
        original_seq_len = 128,
        n_diff_attn_layers = 1,
        pure_attn_ratio = 0.5,
        n_blocks = 5
    )
    model = StateFormer(args)

    print('Model created successfully.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:','cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Muon(model.parameters(), lr=1e-4)
    model.train()

    tokens = torch.randint(0, args.vocab_size, (args.max_batch_size, 129), device=device)
    xtrain, ytrain = tokens[:, :-1], tokens[:, 1:]
    padding_mask = torch.zeros_like(xtrain, device=device, dtype=bool)
    padding_mask[:, :64] = False
    padding_mask[:, 64:] = True
    
    print('Start training...')
    for i in range(20):
        optimizer.zero_grad()
        # print(f'Epoch {i+1} started.')
        logits = model(xtrain, padding_mask = padding_mask, start_pos=0)
        # print(f'Epoch {i+1} forward pass completed.')
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            ytrain.reshape(-1)
        ).float()
        loss.backward()
        # print(f'Epoch {i+1} backward pass completed.')
        optimizer.step()
        # optimizer.zero_grad()
        print(f'Epoch {i+1} completed.')
    print('Model trained successfully.')
    with torch.no_grad():
        print(model(xtrain, padding_mask = padding_mask).to(torch.float16).argmax(dim=-1))
    print(ytrain[:, :64])
    print('Program tested successfully.')

elif '\?' in sys.argv or '-?' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
    print(f'{sys.argv[0]} <command>')
    print(
'''Commands:
    --test    -t    /t : Test program with random tokens
    --help    -?    /? : Show this message
''')

else:
    print('Invalid command.')
    print('Use -? command for help.')
