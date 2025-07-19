
#include <cuda_runtime.h>
#include <math_functions.h>

// f(x) for forward pass
__global__ void f_lt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = x[idx] - 1.0f;
		y[idx] = 1.0f / (t*t);
	}
}

__global__ void f_ge_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = x[idx] + 1.0f;
		y[idx] = t*t;
	}
}

// f(x) for backward pass
__global__ void f_back_lt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f - x[idx];
		y[idx] = 2.0f / (t*t*t);
	}
}

__global__ void f_back_ge_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		y[idx] = 2.0f * x[idx] + 2.0f;
	}
}

// g(x) for forward pass
__global__ void g_gt_0_lt_1(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		y[idx] = 1.0f - rsqrtf(x[idx]);
	}
}

__global__ void g_ge_1(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		y[idx] = sqrtf(x[idx]) - 1.0f;
	}
}

// g(x) for backward pass
__global__ void g_back_gt_0_lt_1(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		y[idx] = 1.0f / (2 * x[idx] * sqrtf(x[idx]));
	}
}

__global__ void g_back_ge_1(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		y[idx] = 1.0f / (2 * sqrtf(x[idx]));
	}
}

// f_sigmoid for forward pass ("sigmoid" function with function f instead of exp)
__global__ void f_sigmoid_le_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f - x[idx];
		y[idx] = 1.0f / (1.0f + t * t);
	}
}

__global__ void f_sigmoid_gt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f + x[idx];
		t *= t;
		y[idx] = t / (t + 1.0f);
	}
}

// f_sigmoid for backward pass ("sigmoid" function with function f instead of exp)
__global__ void f_sigmoid_back_le_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f - x[idx];
		y[idx] = 2.0f * t ;
		t *= t;
		t += 1.0f;
		y[idx] /= (t * t);
	}
}

__global__ void f_sigmoid_back_gt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f + x[idx];
		y[idx] = 2.0f * t;
		t *= t;
		t += 1.0f;
		y[idx] /= (t * t);
	}
}

// f_silu for forward pass ("silu" function with function f instead of exp)
__global__ void f_silu_le_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f - x[idx];
		y[idx] = x[idx] / (1.0f + t * t);
	}
}

__global__ void f_silu_gt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f + x[idx];
		t *= t;
		y[idx] = x[idx] * t / (t + 1.0f);
	}
}

// f_silu for backward pass ("silu" function with function f instead of exp)
__global__ void f_silu_back_le_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f - x[idx];
		t *= t;
		t += 1.0f;
		t *= t;
		y[idx] = 2.0f - (x[idx] * x[idx]);
		y[idx] /= t;
	}
}

__global__ void f_silu_back_gt_0(float* x, float* y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) 
	{
		float t = 1.0f + x[idx];
		t *= t;
		t += 1.0f;
		y[idx] = x[idx] * x[idx] - 2.0f;
		y[idx] /= t * t;
		y[idx] += 1.0f;
	}
}