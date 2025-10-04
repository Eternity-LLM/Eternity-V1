from . import fwd, bwd
import torch

# Note: kernels for backward pass is still developing.

class ActQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, block_size:int = 128, scale_fmt:str):
        y, scale = fwd.act_quant(x, block_size, scale_fmt)
        ctx.save_for_backward(x, scale)
        ctx.block_size = block_size
        ctx.scale_fmt = scale_fmt
        return y, scale

class Fp8GemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a:torch.Tensor, a_s:torch.Tensor, b:torch.Tensor, b_s:torch.Tensor):
        y = fwd.fp8_gemm(a, a_s, b, b_s)
        ctx.save_for_backward(a, a_s, b, b_s)
        return y

class Fp8IndexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q:torch.Tensor, q_s:torch.Tensor, k:torch.Tensor, k_s:torch.Tensor):
        y = fwd.fp8_index(q, q_s, k, k_s)
        ctx.save_for_backward(q, q_s, k, k_s)
        return y