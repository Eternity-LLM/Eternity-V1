#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

// Load functions.ptx
CUmodule functions_module;
std::ifstream file{ "functions.ptx" };
std::string functions_ptx_code(
    (std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());

CUfunction f_lt_0_kernel;
cuModuleGetFunction(&f_lt_0_kernel, functions_module, "f_lt_0");

CUfunction f_ge_0_kernel;
cuModuleGetFunction(&f_lt_0_kernel, functions_module, "f_ge_0");

CUfunction f_back_lt_0_kernel;
cuModuleGetFunction(&f_back_lt_0_kernel, functions_module, "f_back_lt_0");

CUfunction f_back_ge_0_kernel;
cuModuleGetFunction(&f_back_ge_0_kernel, functions_module, "f_back_ge_0");

CUfunction g_gt_0_lt_1_kernel;
cuModuleGetFunction(&g_gt_0_lt_1_kernel, functions_module, "g_gt_0_lt_1");

CUfunction g_ge_1_kernel;
cuModuleGetFunction(&g_ge_1_kernel, functions_module, "g_ge_1");

CUfunction g_back_gt_0_lt_1_kernel;
cuModuleGetFunction(&g_back_gt_0_lt_1_kernel, functions_module, "g_back_gt_0_lt_1");

CUfunction g_back_ge_1_kernel;
cuModuleGetFunction(&g_back_ge_1_kernel, functions_module, "g_back_ge_1");

CUfunction f_sigmoid_le_0_kernel;
cuModuleGetFunction(&f_sigmoid_le_0_kernel, functions_module, "f_sigmoid_le_0");



