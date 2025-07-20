#include <ATen/ATen.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "cpp_extention_ptx_code.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


torch::Tensor f_lt_0(torch::Tensor a, torch::Tensor b)
{

}