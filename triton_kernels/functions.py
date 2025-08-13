# Eternity-V1 

import torch
from .forward_kernels import *
from .backward_kernels import *

class ActQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor):
        # Forward pass for act-quant function
        # Arguments:
        #     x (torch.Tensor): Input tensor
        # Returns:
        #     torch.Tensor: Output tensor after quantization
        y, s = act_quant(x)
        ctx.save_for_backward(x,s)
        return y, s
    
    @staticmethod
    def backward(ctx, grad_y, grad_s):
        # Backward pass for act-quant function
        # Arguments:
        #     grad_y (torch.Tensor): Gradient of the output tensor
        #     grad_s (torch.Tensor): Gradient of the scale factor
        # Returns:
        #     torch.Tensor: Gradient of the input tensor
        x, s = ctx.saved_tensors
        grad_x = act_quant_backward(grad_y, x, s)
        return grad_x

class WeightDequantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, s:torch.Tensor):
        # Forward pass for weight-dequant function
        # Arguments:
        #     x (torch.Tensor): Input tensor
        #     s (torch.Tensor): Scale factor tensor
        # Returns:
        #     torch.Tensor: Output tensor after dequantization
        y = weight_dequant(x, s)
        ctx.save_for_backward(x, s)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        # Backward pass for weight-dequant function
        # Arguments:
        #     grad_y (torch.Tensor): Gradient of the output tensor
        # Returns:
        #     torch.Tensor: Gradient of the input tensor and scale factor
        x, s = ctx.saved_tensors
        grad_x, grad_s = weight_dequant_backward(grad_y, x, s)
        return grad_x, grad_s

class FP8GEMMFunction(torch.autograd.Function):
    # still developing
    pass