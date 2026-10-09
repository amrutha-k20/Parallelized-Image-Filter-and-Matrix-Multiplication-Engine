#pragma once

#include "common/types.hpp"

namespace hpc {

// Phase 3: Naive Global Memory CUDA Implementations
void filter_naive_cuda(const Image& h_input, Image& h_output, const Kernel& kernel, float* out_kernel_ms = nullptr);
void sobel_naive_cuda(const Image& h_input, Image& h_output, float* out_kernel_ms = nullptr);

// Phase 4: Advanced GPU Optimization (Shared Memory Tiling with Halo Management)
void filter_tiled_cuda(const Image& h_input, Image& h_output, const Kernel& kernel, float* out_kernel_ms = nullptr);
void sobel_tiled_cuda(const Image& h_input, Image& h_output, float* out_kernel_ms = nullptr);

} // namespace hpc
