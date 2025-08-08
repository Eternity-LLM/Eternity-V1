# Eternity-V1 modeling.py
# Edited by Haozhe Xu (14), Eternity-LLM Organization
# Based on DeepSeek-V3 (url https://github.com/deepseek-ai/deepseek-v3/)
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
use_deepseek = False # If true, part of the parameters will be fine-tuned from DeepSeek-R1-0528, but model architecture is still Eternity-V1 (hybird model).
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
    # ssa (state space attention)
    q_lora_rank:int = 1536
    kv_lora_rank:int = 512
    qk_nope_head_dim:int = 128
    qk_rope_head_dim:int = 64
    v_head_dim:int = 128
    n_heads:int = 128
    mscale:float = 1.0
    ssa_block_len:int = 64
    # dla (differential latent attention)
    dla_q_lora_rank:int = 1536
    dla_kv_lora_rank:int = 512
    dla_qk_nope_head_dim:int = 128
    dla_qk_rope_head_dim:int = 64
    dla_v_head_dim:int = 128
    dla_n_heads:int = 128
    dla_mscale:float = 1.0
    # ghm (gated hybrid module)
    gate_dim = 4096
    conv_kernel_size:int = 3
    ssm_state_dim:int = 128
    ssm_pe_state_dim:int = 64
    ssm_head_dim:int = 128
    ssm_n_heads:int = 128
    ssm_lora_rank:int = 512
    # mlp (multi-layer perceptrom)
    mlp_dim:int = 18432
    # moe (mixture of experts)
    moe_dim:int = 2048
    n_shared:int = 1
    n_routed:int = 256
    n_routed_group:int = 8
    n_topk_group:int = 4
    n_experts_per_tok:int = 8
    n_dense_layers:int = 3
    routed_scale:float = 2.5
    shared_conv_kernel_size:int = 32
    # rope
    beta_fast:int = 32
    beta_slow:int = 1
    rope_theta:float = 10000.0
    rope_factor:float = 40.0
    original_seq_len:int = 4096
    # block
    n_diff_attn_layers:int = 5
    pure_attn_ratio:float = 0.08
    n_blocks:int = 61

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

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
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

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
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

        self.lambda_init = 0.8 - 0.6 * torch.exp(-0.3 * layer_idx)
        self.lambda_q_nope = nn.Parameter(torch.tensor(self.qk_nope_head_dim))
        self.lambda_q_rope = nn.Parameter(torch.tensor(self.qk_rope_head_dim))
        self.lambda_k_nope = nn.Parameter(torch.tensor(self.qk_nope_head_dim))
        self.lambda_k_rope = nn.Parameter(torch.tensor(self.qk_rope_head_dim))
        self.register_parameter("lambda_", None)
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_head_dim + self.v_head_dim))
        self.scale = self.qk_head_dim ** (-0.5)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.dla_mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]):
        if self.lambda_ is None:
            lambda_ = torch.exp(self.lambda_q_nope @ self.lambda_k_nope) - torch.exp(self.lambda_q_rope @ self.lambda_k_rope) + self.lambda_init
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
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        nope_scores = torch.einsum('bshc,btc->bsht', q_nope, self.kv_cache[:bsz, :end_pos]) * self.scale
        pe_scores = torch.einsum('bshr,btr->bsht', q_pe, self.pe_cache[:bsz, :end_pos]) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            nope_scores += mask
            pe_scores += mask

        nope_weights = u.f_softmax(nope_scores.to(torch.float32), dim=-1)
        pe_weights = u.f_softmax(pe_scores.to(torch.float32), dim=-1)
        weights = nope_weights - lambda_ * pe_weights
        x = torch.einsum('bsht,btc->bshc', weights, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

class SSA(nn.Module):
    # State Space Attention (SSA)
    # Note that this mechanism is equivalent to linear attention  mechanism: $ \bm{Y} = \bm{L} \circ (\bm{Q} \cdot \bm{K}^T) \cdot \bm{V} $ , 
    # where $\circ$ is the element-wise product, $\bm{L}$ is the linear mask, $\bm{Q}$ is the query, $\bm{K}$ is the key, and $\bm{V}$ is the value.
    # I made several changes to the state space dual (SSD) model. For more details of the original SSD model, please read the paper arXiv:2405.21060v1 (Dao and Gu, 2024).
    # still developing, not finished yet.
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
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_head_dim + self.v_head_dim))
        self.scale = self.qk_head_dim ** (-0.5)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.dla_mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        self.kv_states, self.pe_states = None, None
        self.function = u.StateSpaceAttentionFunction()

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, linear_mask:Optional[torch.Tensor]):
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
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        kv = self.wkv_b(kv)
        kv = kv.view(bsz, seqlen, self.n_local_heads, -1)
        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        nope_out, nope_states = self.function.apply(q_nope, k, v, linear_mask, init_states=self.kv_states, block_len=self.block_len)
        pe_out, pe_states = self.function.apply(q_pe, k_pe, v, linear_mask, init_states=self.pe_states, block_len=self.block_len)
        if not self.training:
        	self.kv_states = nope_states
        	self.pe_states = pe_states
        nope_out = nope_out * self.scale
        pe_out = pe_out * self.scale
        output = nope_out + pe_out
        output = self.wo(output.view(bsz, seqlen, -1))
        return output


class ParallelSeperableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, max_batch_size=16) -> None:
        super().__init__()
        assert out_channels % world_size == 0, f"Output channels must be divisible by world size (world_size={world_size})"
        self.part_out_channels = out_channels // world_size
        self.cache = torch.zeros(max_batch_size, in_channels, kernel_size, dtype=torch.bfloat16)
        
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
        x = x.transpose(1, 2)
		cache = torch.cat(self.cache, x)[:, :, 1:]
        if not self.training:
            self.cache = cache
        return self.pointwise(self.depthwise(cache))

class GHM(nn.Module):
    # Gated Hybrid Module (GHM)
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
        self.part_state_dim = args.ssm_state_dim // world_size
        self.part_pe_state_dim = args.ssm_pe_state_dim // world_size
        # self.part_head_dim = args.ssm_head_dim // world_size

        self.attn = SSA(args)
        self.gate_1 = ColumnParallelLinear(args.dim, args.gate_dim)
        self.gate_2 = RowParallelLinear(args.gate_dim, 2)
        self.ssm_linear_proj = Linear(args.dim, args.ssm_lora_rank + args.ssm_head_dim)
        # Order: A, B, C, X
        self.nope_conv = ParallelSeperableConv1d(args.ssm_lora_rank, args.ssm_state_dim * args.ssm_n_heads * 2 + args.ssm_n_heads * args.ssm_head_dim, 
                                         kernel_size=args.conv_kernel_size, max_batch_size=args.max_batch_size)
        self.pe_conv = ParallelSeperableConv1d(args.ssm_lora_rank, args.ssm_pe_state_dim * args.ssm_n_heads * 2,
                                         kernel_size=args.conv_kernel_size, max_batch_size=args.max_batch_size)
        self.ssm_out_proj = RowParallelLinear(args.ssm_n_heads * args.ssm_head_dim, args.dim)
    	

    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor):
        scores = self.gate_2(u.f_silu(self.gate_1(x)))
        scores = u.f_sigmoid(scores)                 # (batch_size, seq_len, 2)
        scores = scores.sum(dim = 1, keepdim=False)  # (batch_size, 2)

        ssa_scores = scores[:, :, :1].squeeze(-1)
        ssm_scores = scores[:, :, 1:].squeeze(-1)

        ssa_idx = ssa_scores >= ssm_scores
        ssm_idx = ssa_scores < ssm_scores

        output = torch.zeros_like(x)
		
        ssa_inputs = x[ssa_idx]
        ssm_inputs = x[ssm_idx]

        nope_states = self.attn.kv_states
        pe_states = self.attn.pe_states

        if ssa_inputs.numel():
			output[ssa_idx] = self.attn(ssa_inputs, start_pos, freqs_cis, None) * ssa_scores[ssa_idx].unsqueeze(-1)
            if not self.training:
            	nope_states[ssa_idx] = self.attn.kv_states
            	pe_states[ssa_idx] = self.attn.pe_states
        if ssm_inputs.numel():
			#h = self.ssm_linear_proj(ssm_inputs)
            h, A = torch.split(self.ssm_linear_proj(ssm_inputs), [self.ssm_lora_rank, self.head_dim], dim=-1)
            nope_inputs = self.nope_conv(h)
            pe_inputs = self.pe_conv(h)

            nope_B, nope_C, X = torch.split(nope_inputs, [self.part_state_dim * self.n_heads, 
                                                        self.part_state_dim * self.n_heads, self.part_head_dim * self.n_heads], dim=-1)
            pe_B, pe_C= torch.split(pe_inputs, [self.part_pe_state_dim * self.n_heads,
												self.part_pe_state_dim * self.n_heads, self.part_head_dim * self.n_heads], dim=-1)
            
            ssm_nope_outputs, ssm_nope_states = u.ssd(X, A, nope_B, nope_C, block_len=self.attn.block_len, initial_states=nope_states)
            ssm_pe_outputs, ssm_pe_states = u.ssd(X, A, pe_B, pe_C, block_len=self.attn.block_len, initial_states=pe_states)
            if not self.training:
                nope_states[ssm_idx] = ssm_nope_states
                pe_states[ssm_idx] = ssm_pe_states
            ssm_outputs = ssm_nope_outputs + ssm_pe_outputs
            ssm_outputs = ssm_outputs * self.attn.scale
            ssm_outputs = self.ssm_out_proj(ssm_outputs.view(ssm_outputs.size(0), ssm_outputs.size(1), -1))
            output[ssm_idx] = ssm_outputs * ssm_scores[ssm_idx].unsqueeze(-1)
        if not self.training:
            self.attn.kv_states = nope_states
            self.attn.pe_states = pe_states
        return output

class MLP(nn.Module):
    # Multi-Layer Perceptron (MLP) with seperable convolutional (opt.)
    def __init__(self, dim:int, mlp_dim:int, use_conv:bool = False, conv_kernel_size:int = 32) -> None:
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, mlp_dim)
        self.w2 = RowParallelLinear(mlp_dim, dim)
        self.w3 = ColumnParallelLinear(dim, mlp_dim)
        self.conv = None
        if use_conv:
            self.conv = ParallelSeperableConv1d(dim, mlp_dim, kernel_size=conv_kernel_size, max_batch_size=16)
    
    def forward(self, x:torch.Tensor):
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
    def forward(self, x:torch.Tensor):
        scores = linear(x, self.weight)
        scores = u.f_sigmoid(scores)
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
        weights *= self.route_scale
        return weights.type_as(x), indices

class Expert(nn.Module):
    def __init__(self, dim:int, moe_dim:int):
        sumer().__init__()
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
        self.shared = MLP(args.dim, args.n_shared * args.moe_dim, use_conv=True, conv_kernel_size = args.shared_conv_kernel_size) if args.n_shared > 0 else None
    
    def forward(self, x:torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed).tolist()
        for i in range(self.st_idx, self.en_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if world_size > 1:
            dist.all_reduce(y)
        if self.shared is not None:
            y += self.shared(x)
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
        self.ffn = MLP(args.dim, args.mlp_dim) if layer_idx < args.n_dense_layers else MoE(args)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)
    def forward(self, x:torch.Tensor, start_pos:int, freqs_cis:torch.Tensor, mask:Optional[torch.Tensor]):
        x = x + self.seq_transformation(self.norm1(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
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
    
    # If you are using this model for inference, then add this line of code:
    # @torch.inference_mode()
    def forward(self, tokens:torch.Tensor, start_pos:int=0, attn_mask:Optional[torch.Tensor]=None, linear_mask:Optional[torch.Tensor]=None) -> None :
        seqlen = tokens.size(1)
        h = self.emb(tokens)
        freqs_cis = self.freqs_cis[start_pos:(start_pos+seqlen)]
        n = 0
        for layer in self.layers:
            if n < self.n_diff_attn_layers:
                h = layer(h, start_pos, freqs_cis, attn_mask)
            else:
                h = layer(h, start_pos, freqs_cis, linear_mask)
            n += 1
        h = self.norm(h)
        logits = self.final(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim = -1)
        return logits