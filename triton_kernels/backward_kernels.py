# Eternity-V1 triton kernels for backward operations

# Edited by  Haozhe Xu and DeepSeek-V3-0324

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config

@triton.jit
def act_quant_backward_kernel(
    grad_y_ptr,  # Input gradient tensor pointer [..., D]
    x_ptr,       # Original input tensor pointer [..., D]
    s_ptr,       # Saved scale factors pointer [..., D//BLOCK_SIZE]
    grad_x_ptr,  # Output gradient tensor pointer [..., D]
    BLOCK_SIZE: tl.constexpr,  # Processing block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < grad_x_ptr.shape[-1]  # Boundary check
    
    # Load data
    grad_y = tl.load(grad_y_ptr + offs, mask=mask)
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(s_ptr + pid)
    
    # Compute gradient
    grad_x = grad_y / s
    grad_x = tl.where(tl.abs(x) <= 448.0 * s, grad_x, 0.0)  # Clip gradient
    
    tl.store(grad_x_ptr + offs, grad_x, mask=mask)

def act_quant_backward(
    grad_y: torch.Tensor,  # Input gradient tensor
    x: torch.Tensor,       # Original input tensor
    s: torch.Tensor,       # Scale factors tensor
    block_size: int = 128  # Processing block size
) -> torch.Tensor:         # Returns gradient w.r.t input
    assert grad_y.is_contiguous() and x.is_contiguous()
    grad_x = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_backward_kernel[grid](grad_y, x, s, grad_x, BLOCK_SIZE=block_size)
    return grad_x

@triton.jit
def weight_dequant_backward_kernel(
    grad_y_ptr,  # Input gradient pointer [M, N]
    x_ptr,       # Quantized weight pointer [M, N]
    s_ptr,       # Scale factors pointer [M, N//BLOCK_SIZE]
    grad_x_ptr,  # Output gradient (dL/dx) pointer [M, N]
    grad_s_ptr,  # Output gradient (dL/ds) pointer [M, N//BLOCK_SIZE]
    M, N,        # Matrix dimensions
    BLOCK_SIZE: tl.constexpr,  # Processing block size
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load data
    grad_y = tl.load(grad_y_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    s = tl.load(s_ptr + pid_m * (N // BLOCK_SIZE) + pid_n)
    
    # Compute gradients
    grad_x = grad_y * s
    grad_s_block = tl.sum(grad_y * x)  # Reduce gradient per block
    
    # Store results
    tl.store(grad_x_ptr + offs_m[:, None] * N + offs_n[None, :], grad_x, mask=mask)
    tl.store(grad_s_ptr + pid_m * (N // BLOCK_SIZE) + pid_n, grad_s_block)

def weight_dequant_backward(
    grad_y: torch.Tensor,  # Input gradient tensor
    x: torch.Tensor,       # Quantized weight tensor
    s: torch.Tensor,       # Scale factors tensor
    block_size: int = 128  # Processing block size
) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns (dL/dx, dL/ds)
    assert grad_y.is_contiguous() and x.is_contiguous()
    M, N = x.shape
    grad_x = torch.empty_like(x)
    grad_s = torch.zeros_like(s)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_backward_kernel[grid](grad_y, x, s, grad_x, grad_s, M, N, BLOCK_SIZE=block_size)
    return grad_x, grad_s

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_backward_kernel(
    grad_c_ptr,  # Input gradient pointer [M, N]
    a_ptr, b_ptr,  # Forward input pointers [M, K], [K, N]
    a_s_ptr, b_s_ptr,  # Scale factor pointers [M, K//BLK_K], [N//BLK_K, K]
    grad_a_ptr, grad_b_ptr,  # Output gradient pointers [M, K], [K, N]
    grad_a_s_ptr, grad_b_s_ptr,  # Scale gradient pointers
    M, N: tl.constexpr, K: tl.constexpr,  # Matrix dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Tiling size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Tiling size for N dimension
    BLOCK_SIZE_K: tl.constexpr,  # Tiling size for K dimension
):
    # Compute dL/dA --------------------------------------------------
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    
    a_s = tl.load(a_s_ptr + pid_m * (K // BLOCK_SIZE_K) + pid_k)
    acc_a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n in range(0, N, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_b = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        grad_c = tl.load(grad_c_ptr + offs_m[:, None] * N + offs_n[None, :], 
                        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_b)
        b_s = tl.load(b_s_ptr + (offs_n // BLOCK_SIZE_K)[None, :] * (K // BLOCK_SIZE_K) + pid_k)
        
        acc_a += tl.dot(grad_c, b.T) * a_s * b_s
    
    tl.store(grad_a_ptr + offs_m[:, None] * K + offs_k[None, :], acc_a, mask=mask_a)
    
    # Compute dL/dB --------------------------------------------------
    pid_n = tl.program_id(2)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    
    b_s = tl.load(b_s_ptr + pid_n * (K // BLOCK_SIZE_K) + pid_k)
    acc_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m in range(0, M, BLOCK_SIZE_M):
        offs_m = m + tl.arange(0, BLOCK_SIZE_M)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_a)
        a_s = tl.load(a_s_ptr + offs_m * (K // BLOCK_SIZE_K) + pid_k)
        grad_c = tl.load(grad_c_ptr + offs_m[:, None] * N + offs_n[None, :], 
                        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        
        acc_b += tl.dot(a.T, grad_c) * a_s * b_s
    
    tl.store(grad_b_ptr + offs_k[:, None] * N + offs_n[None, :], acc_b, mask=mask_b)
    
    # Compute scale gradients ----------------------------------------
    if pid_k == 0:
        grad_a_s = tl.sum(acc_a * tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_a))
        tl.store(grad_a_s_ptr + pid_m * (K // BLOCK_SIZE_K) + pid_k, grad_a_s)

def fp8_gemm_backward(
    grad_c: torch.Tensor,  # Input gradient tensor
    a: torch.Tensor,       # Forward input A
    b: torch.Tensor,       # Forward input B
    a_s: torch.Tensor,     # Scale factors for A
    b_s: torch.Tensor      # Scale factors for B
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # Returns (dL/dA, dL/dB, dL/dA_s, dL/dB_s)
    assert all(t.is_contiguous() for t in [grad_c, a, b, a_s, b_s])
    M, K = a.shape
    N = b.shape[1]
    
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    grad_a_s = torch.zeros_like(a_s)
    grad_b_s = torch.zeros_like(b_s)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    fp8_gemm_backward_kernel[grid](
        grad_c, a, b, a_s, b_s,
        grad_a, grad_b, grad_a_s, grad_b_s,
        M, N, K
    )
    return grad_a, grad_b, grad_a_s, grad_b_s

