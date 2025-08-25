# Eternity-V1 modeling.py
# Edited by Haozhe Xu (14), Eternity-LLM Organization
# Based on DeepSeek-V3 (url https://github.com/deepseek-ai/deepseek-v3/)
# Note that we use hybrid model (Attention, MLP, MoE, SSM)

from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from utils import act_quant, weight_dequant, fp8_gemm
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
        # embedding
        dim: int = 7168,
        vocab_size: int = 129280,
        emb_lora_rank: int = 256,
        # sequence transformations
        # ssa (state space attention)
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        n_heads: int = 128,
        mscale: float = 1.0,
        ssa_block_len: int = 64,
        # dla (differential latent attention)
        dla_q_lora_rank: int = 1536,
        dla_kv_lora_rank: int = 512,
        dla_mscale: float = 1.0,
        dla_max_attn_score: float = 100.0,
        # ghm (gated hybrid module)
        gate_dim: int = 4096,
        conv_kernel_size: int = 3,
        ssm_lora_rank: int = 512,
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

        # block
        n_diff_attn_layers: int = 5,
        pure_attn_ratio: float = 0.08,
        n_blocks: int = 61
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.dim = dim
        self.vocab_size = vocab_size
        self.emb_lora_rank = emb_lora_rank
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_heads = n_heads
        self.mscale = mscale
        self.ssa_block_len = ssa_block_len
        self.dla_q_lora_rank = dla_q_lora_rank
        self.dla_kv_lora_rank = dla_kv_lora_rank
        self.dla_qk_nope_head_dim = qk_nope_head_dim
        self.dla_qk_rope_head_dim = qk_rope_head_dim
        self.dla_v_head_dim = v_head_dim
        self.dla_n_heads = n_heads
        self.dla_mscale = dla_mscale
        self.dla_max_attn_score = dla_max_attn_score
        self.gate_dim = gate_dim
        self.conv_kernel_size = conv_kernel_size
        self.ssm_state_dim = qk_nope_head_dim
        self.ssm_pe_state_dim = qk_rope_head_dim
        self.ssm_head_dim = v_head_dim
        self.ssm_n_heads = n_heads
        self.ssm_lora_rank = ssm_lora_rank
        self.mlp_dim = mlp_dim
        self.mlp_conv_kernel_size = mlp_conv_kernel_size
        self.moe_dim = moe_dim
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.n_routed_group = n_routed_group
        self.n_topk_group = n_topk_group
        self.n_experts_per_tok = n_experts_per_tok
        self.n_dense_layers = n_dense_layers
        self.routed_scale = routed_scale
        self.shared_conv_kernel_size = shared_conv_kernel_size
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.original_seq_len = original_seq_len
        self.n_diff_attn_layers = n_diff_attn_layers
        self.pure_attn_ratio = pure_attn_ratio
        self.n_blocks = n_blocks

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

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, use_scale:bool=True) -> torch.Tensor:
    # Applies a linear transformation to the incoming data: $y=xA^T+b$
    # This function supports specialized implementations based on quantization
    # and tensor formats
    # Args:
    #     x (torch.Tensor) : The input tensor
    #     weight (torch.Tensor) : The weight tensor which may be quantized and requires
    #                             dequantization for certain cases
    #     bias (Optional[torch.Tensor]): The bias tensor to be added, default is None
    # Returns:
    #     torch.Tensor: The output tensor after applying the linear transformation
    
    if weight.element_size() > 1 or not use_scale:
        x = x.to(weight.dtype)
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, use_scale:bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1 and use_scale:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.use_scale = use_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.use_scale)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, use_scale:bool = True):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype, use_scale)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, use_scale:bool = True):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype, use_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, use_scale=self.use_scale)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x:torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

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

class DLA(nn.Module):
    # Differential Latent Attention (DLA) 
    def __init__(self, args:ModelArgs, layer_idx:int) -> None:
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.dla_n_heads
        self.n_local_heads = args.dla_n_heads // world_size
        self.q_lora_rank = args.dla_q_lora_rank
        self.kv_lora_rank = args.dla_kv_lora_rank
        self.qk_nope_head_dim = args.dla_qk_nope_head_dim
        self.qk_rope_head_dim = args.dla_qk_rope_head_dim
        self.v_head_dim = args.dla_v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * layer_idx))
        self.lambda_q_nope = nn.Parameter(torch.zeros(self.qk_nope_head_dim))
        self.lambda_q_rope = nn.Parameter(torch.zeros(self.qk_rope_head_dim))
        self.lambda_k_nope = nn.Parameter(torch.zeros(self.qk_nope_head_dim))
        self.lambda_k_rope = nn.Parameter(torch.zeros(self.qk_rope_head_dim))
        self.register_parameter("lambda_", None)
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
            mscale = 0.1 * args.dla_mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        if attn_impl == 'naive':
            self.register_buffer('k_cache_1', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim//2), persistent=False)
            self.register_buffer('k_cache_2', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim//2), persistent=False)
            self.register_buffer('v_cache', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

        self.max_attn_score = args.dla_max_attn_score

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor] = None):
        if self.training:
            self.lambda_ = None
        if self.lambda_ is None:
            lambda_ = torch.exp(self.lambda_q_nope @ self.lambda_k_nope) - torch.exp(self.lambda_q_rope @ self.lambda_k_rope) + self.lambda_init
            if not self.training:
                self.lambda_ = nn.Parameter(lambda_)
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.wq_a(x))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == 'naive':
            q_nope_1, q_nope_2 = torch.chunk(q_nope, 2, dim=-1)
            q_pe_1, q_pe_2 = torch.chunk(q_pe, 2, dim=-1)
            q_1, q_2 = torch.cat((q_nope_1, q_pe_1), dim=-1), torch.cat((q_nope_2, q_pe_2), dim=-1)

            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_nope_1, k_nope_2 = torch.chunk(k_nope, 2, dim=-1)
            k_pe_1, k_pe_2 = torch.chunk(k_pe, 2, dim=-1)

            k_1 = torch.cat((k_nope_1, k_pe_1.expand(-1, -1, self.n_local_heads, -1)), dim=-1)
            k_2 = torch.cat((k_nope_2, k_pe_2.expand(-1, -1, self.n_local_heads, -1)), dim=-1)

            self.k_cache_1[:bsz, start_pos:end_pos] = k_1
            self.k_cache_2[:bsz, start_pos:end_pos] = k_2
            self.v_cache[:bsz, start_pos:end_pos] = v

            scores_1 = torch.einsum('bshd,bthd->bsht', q_1.to(torch.float32), self.k_cache_1[:bsz, :end_pos].to(torch.float32)) * self.scale
            scores_2 = torch.einsum('bshd,bthd->bsht', q_2.to(torch.float32), self.k_cache_2[:bsz, :end_pos].to(torch.float32)) * self.scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = kv
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            
            q_nope_1, q_nope_2 = torch.chunk(q_nope, 2, dim=-1)
            q_pe_1, q_pe_2 = torch.chunk(q_pe, 2, dim=-1)

            k_nope_1, k_nope_2 = torch.chunk(self.kv_cache[:bsz, :end_pos], 2, dim = -1)
            k_pe_1, k_pe_2 = torch.chunk(self.pe_cache[:bsz, :end_pos], 2, dim = -1)

            scores_1 = (torch.einsum('bshc,btc->bsht', q_nope_1.to(torch.float32), k_nope_1.to(torch.float32)) + \
                    torch.einsum('bshr,btr->bsht', q_pe_1.to(torch.float32), k_pe_1.to(torch.float32))) * self.scale
            
            scores_2 = (torch.einsum('bshc,btc->bsht', q_nope_2.to(torch.float32), k_nope_2.to(torch.float32)) + \
                    torch.einsum('bshr,btr->bsht', q_pe_2.to(torch.float32), k_pe_2.to(torch.float32))) * self.scale

		# QK-Clip. See Kimi-K2: Open Agentic Intelligence (arXiv:2507.20534)
        if self.training:
            max_1 = scores_1.transpose(-3, -2).view(bsz, self.n_local_heads, -1).max(dim=-1)[0]
            max_2 = scores_2.transpose(-3, -2).view(bsz, self.n_local_heads, -1).max(dim=-1)[0]

            max_1[max_1==0] += 1e-6
            max_2[max_2==0] += 1e-6

            eta_1 = torch.minimum(self.max_attn_score / max_1, torch.tensor(1.0, device=scores_1.device, dtype=scores_1.dtype))
            eta_2 = torch.minimum(self.max_attn_score / max_2, torch.tensor(1.0, device=scores_2.device, dtype=scores_2.dtype))

            scores_1 = torch.einsum('bh,bsht->bsht', eta_1.to(scores_1.dtype), scores_1)
            scores_2 = torch.einsum('bh,bsht->bsht', eta_2.to(scores_2.dtype), scores_2)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores_1 += mask
            scores_2 += mask

        weights_1 = u.f_softmax(scores_1.to(torch.float32), dim=-1)
        weights_2 = u.f_softmax(scores_2.to(torch.float32), dim=-1)
        weights = weights_1 - lambda_ * weights_2

        if attn_impl == 'naive':
            x = torch.einsum('bsht,bthd->bshd', weights, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum('bsht,btc->bshc', weights, self.kv_cache[:bsz, :end_pos].to(torch.float32))
            x = torch.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.v_head_dim:].to(torch.float32))
        x = self.wo(x.flatten(2))

        if self.training:
            if attn_impl == 'naive':
                self.k_cache_1 = torch.zeros_like(self.k_cache_1)
                self.k_cache_2 = torch.zeros_like(self.k_cache_2)
                self.v_cache = torch.zeros_like(self.v_cache)
            else:
                self.kv_cache = torch.zeros_like(self.kv_cache)
                self.pe_cache = torch.zeros_like(self.pe_cache)
        return x

class SSA(nn.Module):
    # State Space Attention (SSA)
    # Note that this mechanism is equivalent to linear attention  mechanism: $ \bm{Y} = \bm{L} \circ (\bm{Q} \cdot \bm{K}^T) \cdot \bm{V} $ , 
    # where $\circ$ is the element-wise product, $\bm{L}$ is the linear mask, $\bm{Q}$ is the query, $\bm{K}$ is the key, and $\bm{V}$ is the value.
    # I made several changes to the state space dual (SSD) model. For more details of the original SSD model, please read the paper arXiv:2405.21060v1 (Dao and Gu, 2024).
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.block_len = args.ssa_block_len
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
            mscale = 0.1 * args.dla_mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        #self.kv_states, self.pe_states = None, None
        if attn_impl == 'naive':
            self.register_buffer('kv_states', torch.zeros(args.max_batch_size, 1, self.n_local_heads, self.qk_head_dim, self.v_head_dim), persistent=False)
        else:
            self.register_buffer('kv_states', torch.zeros(args.max_batch_size, 1, self.n_local_heads, self.qk_nope_head_dim, self.v_head_dim), persistent=False)
            self.register_buffer('pe_states', torch.zeros(args.max_batch_size, 1, self.n_local_heads, self.qk_rope_head_dim, self.v_head_dim), persistent=False)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
        bsz, seqlen, _ = x.size()
        # end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.wq_a(x))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), freqs_cis)
        kv = self.wkv_b(kv)
        kv = kv.view(bsz, seqlen, self.n_local_heads, -1)
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_nope = u.f_silu(q_nope)
        q_pe = u.f_silu(q_pe)
        k = u.f_silu(k)
        k_pe = u.f_silu(k_pe)

        # Q-Clip and K-Clip inspired by Kimi-K2. For more information about the QK-Clip algorithm, see arXiv:2507.20534
        max_q_nope = q_nope.view(bsz, self.n_local_heads, -1).max(dim=-1)[0]
        max_k_nope = k.view(bsz, self.n_local_heads, -1).max(dim=-1)[0]
        max_q_pe = q_pe.view(bsz, self.n_local_heads, -1).max(dim=-1)[0]
        max_k_pe = k_pe.view(bsz, self.n_local_heads, -1).max(dim=-1)[0]

        max_q_nope[max_q_nope==0] += 1e-6
        max_k_nope[max_k_nope==0] += 1e-6
        max_q_pe[max_q_pe==0] += 1e-6
        max_k_pe[max_k_pe==0] += 1e-6

        eta_q_nope = torch.minimum(10.0 / max_q_nope, torch.tensor(1.0, device=q_nope.device, dtype=q_nope.dtype))
        eta_k_nope = torch.minimum(10.0 / max_k_nope, torch.tensor(1.0, device=k.device, dtype=k.dtype))
        eta_q_pe = torch.minimum(10.0 / max_q_pe, torch.tensor(1.0, device=q_pe.device, dtype=q_pe.dtype))
        eta_k_pe = torch.minimum(10.0 / max_k_pe, torch.tensor(1.0, device=k_pe.device, dtype=k_pe.dtype))

        q_nope = torch.einsum('bh,bshd->bshd', eta_q_nope, q_nope)
        k = torch.einsum('bh,bthd->bthd', eta_k_nope, k)
        q_pe = torch.einsum('bh,bshd->bshd', eta_q_pe, q_pe)
        k_pe = torch.einsum('bh,bthd->bthd', eta_k_pe, k_pe)

        if attn_impl == 'naive':
            q = torch.cat((q_nope, q_pe), dim=-1)
            k = torch.cat((k, k_pe.expand(-1, -1, self.n_local_heads, -1)), dim=-1)
            output, states = u.ssa(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), initial_states=self.kv_states, block_len=self.block_len)
        else:
            nope_out, nope_states = u.ssa(q_nope.to(torch.float32), k.to(torch.float32), v.to(torch.float32), initial_states=self.kv_states, block_len=self.block_len)
            pe_out, pe_states = u.ssa(q_pe.to(torch.float32), k_pe.to(torch.float32), v.to(torch.float32), initial_states=self.pe_states, block_len=self.block_len)
        if not self.training:
            if attn_impl == 'naive':
                self.kv_states = states
            else:
                self.kv_states = nope_states
                self.pe_states = pe_states
        if attn_impl == 'absorb':
            output = nope_out + pe_out
        output = output * self.scale
        output = self.wo(output.view(bsz, seqlen, -1))
        return output

class ParallelSeperableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, max_batch_size=16) -> None:
        super().__init__()
        assert out_channels % world_size == 0, f"Output channels must be divisible by world size (world_size={world_size})"
        self.part_out_channels = out_channels // world_size
        #self.cache = torch.zeros(max_batch_size, in_channels, kernel_size, dtype=torch.bfloat16)
        self.register_buffer('cache', torch.zeros(max_batch_size, in_channels, kernel_size, dtype=torch.bfloat16), persistent=False)
        self.bsz = args.max_batch_size

        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  
            bias=bias
        )
        
        self.pointwise = nn.Conv1d(
            in_channels, 
            self.part_out_channels, 
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        x = x.transpose(-1, -2)
        bsz, _1, _2 = x.size()

        cache = self.cache[:bsz]
        cache = torch.cat((cache, x), dim=-1)[:, :, 1:]
        if not self.training:
            self.cache = cache
        return self.pointwise(self.depthwise(cache)).transpose(-1, -2)  # (batch_size, seq_len, out_channels)

class GHM(nn.Module):
    # Gated Hybrid Module (GHM)
    # This module conbines State Space Attention (SSA) and State Space Model (SSM) with a gating mechanism.
    # The SSM part is based on the SSD model and Mamba-2 Architecture (arXiv:2405.21060v1, Dao and Gu, 2024)
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()

        assert args.ssm_n_heads % world_size == 0, f"Number of SSM heads must be divisible by world size (world_size={world_size})"
        assert args.ssm_state_dim % world_size == 0, f"SSM state dimension must be divisible by world size (world_size={world_size})"
        assert args.ssm_pe_state_dim % world_size == 0, f"SSM PE state dimension must be divisible by world size (world_size={world_size})"
        assert args.ssm_head_dim % world_size == 0, f"SSM head dimension must be divisible by world size (world_size={world_size})"

        self.n_heads = args.ssm_n_heads
        self.state_dim = args.ssm_state_dim
        self.pe_state_dim = args.ssm_pe_state_dim
        self.head_dim = args.ssm_head_dim

        self.part_n_heads = args.ssm_n_heads // world_size
        # self.part_state_dim = args.ssm_state_dim // world_size
        # self.part_pe_state_dim = args.ssm_pe_state_dim // world_size
        # self.part_head_dim = args.ssm_head_dim // world_size
        self.ssm_lora_rank = args.ssm_lora_rank

        self.attn = SSA(args)
        self.gate_1 = ColumnParallelLinear(args.dim, args.gate_dim)
        self.gate_2 = RowParallelLinear(args.gate_dim, 2)
        self.ssm_linear_proj = Linear(args.dim, args.ssm_lora_rank + args.ssm_head_dim)
        # Order: A, B, C, X
        self.conv = ParallelSeperableConv1d(
            args.ssm_lora_rank,
            (args.ssm_state_dim + args.ssm_pe_state_dim) * args.ssm_n_heads * 2 + args.ssm_n_heads * args.ssm_head_dim,
            kernel_size=args.conv_kernel_size,
            max_batch_size=args.max_batch_size
        )
        self.ssm_out_proj = RowParallelLinear(args.ssm_n_heads * args.ssm_head_dim, args.dim)
    	

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
        bsz, _, _ = x.size()
        
        scores = self.gate_2(u.f_silu(self.gate_1(x)))
        scores = u.f_sigmoid(scores)                 # (batch_size, seq_len, 2)
        scores = scores.sum(dim = 1, keepdim=False)  # (batch_size, 2)

        ssa_scores = scores[:, :1].squeeze(-1)
        ssm_scores = scores[:, 1:].squeeze(-1)

        ssa_idx = ssa_scores >= ssm_scores
        ssm_idx = ssa_scores < ssm_scores

        output = torch.zeros_like(x, dtype=torch.bfloat16)
		
        ssa_inputs = x[ssa_idx]
        ssm_inputs = x[ssm_idx]

        if attn_impl == 'naive':
            states = self.attn.kv_states
        else:
            nope_states = self.attn.kv_states
            pe_states = self.attn.pe_states

        if ssa_inputs.numel():
            if attn_impl == 'naive':
                ssa_states = states[ssa_idx] if states is not None else None
                self.attn.kv_states = ssa_states
            else:
                ssa_nope_states = nope_states[ssa_idx] if nope_states is not None else None
                ssa_pe_states = pe_states[ssa_idx] if pe_states is not None else None
                self.attn.kv_states = ssa_nope_states
                self.attn.pe_states = ssa_pe_states
            output[ssa_idx] = torch.einsum('bld,b->bld', self.attn(ssa_inputs, start_pos, freqs_cis), ssa_scores)
            if not self.training:
                if attn_impl == 'naive':
                    if states is None:
                        states = torch.zeros(bsz, self.part_n_heads, self.state_dim+self.pe_state_dim, self.head_dim)
                    states[ssa_idx] = self.attn.kv_states
                else:
                    if nope_states is None:
                        nope_states = torch.zeros(bsz, self.part_n_heads, self.state_dim, self.head_dim)
                    if pe_states is None:
                        pe_states = torch.zeros(bsz, self.part_n_heads, self.pe_state_dim, self.head_dim)
                    nope_states[ssa_idx] = self.attn.kv_states
                    pe_states[ssa_idx] = self.attn.pe_states
        
        if ssm_inputs.numel():
            h, A = torch.split(self.ssm_linear_proj(ssm_inputs), [self.ssm_lora_rank, self.head_dim], dim=-1)
            inputs = self.conv(h)
            bc, X = torch.split(inputs, [-1, self.part_n_heads * self.head_dim])
            
            nope_B, nope_C, pe_B, pe_C = torch.split(bc, [
                self.part_n_heads*self.state_dim,     # nope_B
                self.part_n_heads*self.state_dim,     # nope_C
                self.part_n_heads*self.pe_state_dim,  # pe_B
                self.part_n_heads*self.pe_state_dim   # pe_C
            ])
            '''nope_inputs = self.nope_conv(h)
            pe_inputs = self.pe_conv(h)

            nope_B, nope_C, X = torch.split(nope_inputs, [self.state_dim * self.part_n_heads, 
                                                        self.state_dim * self.part_n_heads, self.head_dim * self.part_n_heads], dim=-1)
            pe_B, pe_C= torch.split(pe_inputs, [self.pe_state_dim * self.part_n_heads,
												self.pe_state_dim * self.part_n_heads, self.head_dim * self.part_n_heads], dim=-1)
            '''
            pe_B = apply_rope(pe_B, freqs_cis)
            pe_C = apply_rope(pe_C, freqs_cis)

            # C-Clip and B-Clip inspired by Kimi-K2. For more information about the QK-Clip algorithm, see arXiv:2507.20534
            max_C_nope = nope_C.view(bsz, self.part_n_heads, -1).max(dim=-1)[0]
            max_B_nope = nope_B.view(bsz, self.part_n_heads, -1).max(dim=-1)[0]
            max_C_pe = pe_C.view(bsz, self.part_n_heads, -1).max(dim=-1)[0]
            max_B_pe = pe_B.view(bsz, self.part_n_heads, -1).max(dim=-1)[0]

            max_C_nope[max_C_nope==0] += 1e-6
            max_B_nope[max_B_nope==0] += 1e-6
            max_C_pe[max_C_pe==0] += 1e-6
            max_B_pe[max_B_pe==0] += 1e-6

            eta_C_nope = torch.minimum(10.0 / max_C_nope, torch.tensor(1.0, device=nope_C.device, dtype=nope_C.dtype))
            eta_B_nope = torch.minimum(10.0 / max_B_nope, torch.tensor(1.0, device=nope_B.device, dtype=nope_B.dtype))
            eta_C_pe = torch.minimum(10.0 / max_C_pe, torch.tensor(1.0, device=pe_C.device, dtype=pe_C.dtype))
            eta_B_pe = torch.minimum(10.0 / max_B_pe, torch.tensor(1.0, device=pe_B.device, dtype=pe_B.dtype))

            nope_C = torch.einsum('bh,bshd->bshd', eta_C_nope, nope_C)
            nope_B = torch.einsum('bh,bthd->bthd', eta_B_nope, nope_B)
            pe_C = torch.einsum('bh,bshd->bshd', eta_C_pe, pe_C)
            pe_B = torch.einsum('bh,bthd->bthd', eta_B_pe, pe_B)

            if attn_impl == 'naive':
                ssm_states = states[ssm_idx]
                B = torch.cat((nope_B, pe_B), dim=-1)
                C = torch.cat((nope_C, pe_C), dim=-1)
                B, C = u.f_silu(B), u.f_silu(C)
                X, A, B, C = [xx.to(torch.float32) for xx in (X, A, B, C)]
                ssm_outputs, ssm_new_states = u.ssd(X, A, B, C, block_len=self.attn.block_len, initial_states=ssm_states)
            else:
                ssm_nope_states = nope_states[ssm_idx] if nope_states is not None else None
                ssm_pe_states = pe_states[ssm_idx] if pe_states is not None else None

                pe_B, pe_C, nope_B, nope_C = [u.f_silu(xx) for xx in (pe_B, pe_C, nope_B, nope_C)]
                
                X, A, nope_B, nope_C, pe_B, pe_C = [xx.to(torch.float32) for xx in (X, A, nope_B, nope_C, pe_B, pe_C)]

                ssm_nope_outputs, ssm_nope_states = u.ssd(X, A, nope_B, nope_C, block_len=self.attn.block_len, initial_states=ssm_nope_states)
                ssm_pe_outputs, ssm_pe_states = u.ssd(X, A, pe_B, pe_C, block_len=self.attn.block_len, initial_states=ssm_pe_states)
            if not self.training:
                if attn_impl == 'naive':
                    states[ssm_idx] = ssm_new_states
                else:
                    nope_states[ssm_idx] = ssm_nope_states
                    pe_states[ssm_idx] = ssm_pe_states
            if attn_impl == 'absorb':
                ssm_outputs = ssm_nope_outputs + ssm_pe_outputs
            ssm_outputs = ssm_outputs * self.attn.scale
            ssm_outputs = self.ssm_out_proj(ssm_outputs.view(ssm_outputs.size(0), ssm_outputs.size(1), -1))
            output[ssm_idx] = torch.einsum('bld,b->bld', ssm_outputs, ssm_scores)
        if not self.training:
            if attn_impl == 'naive':
                self.attn.kv_states = states
            else:
                self.attn.kv_states = nope_states
                self.attn.pe_states = pe_states
        return output

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
    
    def forward(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor] = None):
        h = u.f_silu(self.w1(x)) * self.w3(x)
        if self.conv is not None:
            h += self.conv(x)
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
    def forward(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor]=None):
        scores = linear(x, self.weight)
        scores = u.f_sigmoid(scores)
        if padding_mask is not None:
            scores[padding_mask] = 0
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
        return self.w2(u.f_silu(self.w1(x)) * self.w3(x))

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
    
    def forward(self, x:torch.Tensor, padding_mask:Optional[torch.Tensor]=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, padding_mask)
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
        if layer_idx < args.n_diff_attn_layers:
            self.seq_transformation = DLA(args, layer_idx)
        else:
            if (layer_idx - args.n_diff_attn_layers + 1) % (1 // args.pure_attn_ratio) == 0:
                self.seq_transformation = SSA(args)
            else:
                self.seq_transformation = GHM(args)
        self.ffn = MLP(args.dim, args.mlp_dim, max_batch_size=args.max_batch_size, use_conv=True, conv_kernel_size=args.mlp_conv_kernel_size) if layer_idx < args.n_dense_layers else MoE(args)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)
        self.dr = nn.Dropout(p=0.2)
    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]=None, padding_mask:Optional[torch.Tensor]=None):
        # Arguments:
        #     x: (batch_size, seq_len, dim) tensor of embedded tokens.
        #     start_pos: starting position for the sequence (default: 0).
        #     freqs_cis: precomputed freqs_cis of rotary position embedding (RoPE).
        #     mask: optional attention mask for differential attention layers (default: None).
        #     padding_mask: optional padding mask for the input tokens, with 1s for padding tokens and 0s for non-padding tokens (default: None).
        x = self.dr(x + self.seq_transformation(self.norm1(x), start_pos, freqs_cis, mask) if mask is not None else  \
            x + self.seq_transformation(self.norm1(x), start_pos, freqs_cis))
        x = self.dr(x + self.ffn(self.norm2(x)) if padding_mask is not None else \
            x + self.ffn(self.norm2(x), padding_mask))
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
        self.n_diff_attn_layers = args.n_diff_attn_layers
        self.ratio = args.pure_attn_ratio
        for idx in range(args.n_blocks):
            self.layers.append(Block(args, idx))
        self.norm = RMSNorm(args.dim)
        self.final = ColumnParallelLinear(args.dim, args.vocab_size, dtype = torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        # print(self.freqs_cis.size())
    
    # If you are using this model for inference, then add this line of code:
    # @torch.inference_mode()
    def forward(self, tokens:torch.Tensor, start_pos:int=0, attn_mask:Optional[torch.Tensor]=None, padding_mask:Optional[torch.Tensor]=None) -> None :
        # Eternity-V1 forward pass
        # Arguments:
        #   tokens: (batch_size, seq_len) tensor of tokens.
        #   start_pos: starting position for the sequence (default: 0).
        #   attn_mask: optional attention mask for differential attention layers (default: None).
        #   padding_mask: optional padding mask for the input tokens, with 1s for padding tokens and 0s for non-padding tokens (default: None).
        # Outputs:
        #   logits: (batch_size, seq_len, vocab_size) tensor of logits for each token in the sequence.
        
        seqlen = tokens.size(1)
        h = self.emb(tokens, padding_mask)
        # print('emb')
        freqs_cis = self.freqs_cis[start_pos:(start_pos+seqlen)]
        n = 0
        if attn_mask is None and seqlen >1:
            attn_mask = torch.full((seqlen, seqlen), float('-inf'), device=tokens.device, requires_grad=False).triu_(1)
        for layer in self.layers:
            if n < self.n_diff_attn_layers:
                h = layer(h, start_pos, freqs_cis, attn_mask)
            else:
                h = layer(h, start_pos, freqs_cis)
            n += 1
            # print(n)
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
def cold_start(model:StateFormer):
    def run_cold_start(func):
        def set_requires_grad(r:bool):
            # Set embedding layer
            model.emb.weight.requires_grad = False
            for layer in model.layers:
                seqt = layer.seq_transformation
                ffn = layer.ffn
                # Set attn layers
                attn = seqt if type(seqt).__name__ != 'GHM' else seqt.attn
                if attn.q_lora_rank == 0:
                    attn.wq.weight.requires_grad = r
                else:
                    attn.wq_a.weight.requires_grad = r
                    attn.wq_b.weight.requires_grad = r
                attn.wkv_a.weight.requires_grad = r
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

        def run_func(*args, **kwargs):
            set_requires_grad(False)
            return_val = func(*args, **kwargs)
            set_requires_grad(True)
            return return_val
        return run_func
    return run_cold_start


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
        dla_q_lora_rank = 64,
        dla_kv_lora_rank = 32,
        dla_mscale = 1.0,
        dla_max_attn_score = 100.0,
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
    padding_mask = torch.zeros_like(xtrain, device=device)
    
    print('Start training...')
    for i in range(20):
        optimizer.zero_grad()
        # print(f'Epoch {i+1} started.')
        logits = model(xtrain,padding_mask = padding_mask, start_pos=0)
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
        print(model(xtrain, padding_mask = padding_mask).to(torch.float16))
    print(ytrain)
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