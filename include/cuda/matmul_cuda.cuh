#pragma once

#include "common/types.hpp"

namespace hpc {

// Phase 3: Naive CUDA GEMM
void matmul_naive_cuda(const Matrix& h_A, const Matrix& h_B, Matrix& h_C, float* out_kernel_ms = nullptr);

// Phase 4: Shared Memory Tiled CUDA GEMM
void matmul_tiled_cuda(const Matrix& h_A, const Matrix& h_B, Matrix& h_C, float* out_kernel_ms = nullptr);

} // namespace hpc
