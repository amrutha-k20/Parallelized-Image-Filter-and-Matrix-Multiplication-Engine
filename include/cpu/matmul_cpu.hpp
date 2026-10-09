#pragma once

#include "common/types.hpp"

namespace hpc {

// Phase 1: Sequential Matrix Multiplication (Naive i-j-k baseline)
void matmul_sequential(const Matrix& A, const Matrix& B, Matrix& C);

// Phase 2: OpenMP Parallelized & Cache Locality Optimizations
void matmul_openmp(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_cache_friendly(const Matrix& A, const Matrix& B, Matrix& C);
void matmul_openmp_blocked(const Matrix& A, const Matrix& B, Matrix& C, int blockSize = 64);

} // namespace hpc
