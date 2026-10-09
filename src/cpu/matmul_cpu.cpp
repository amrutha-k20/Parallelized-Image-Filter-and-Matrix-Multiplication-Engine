#include "cpu/matmul_cpu.hpp"
#include <stdexcept>

namespace hpc {

// Sequential Naive GEMM: C = A * B
// Traversal order: i -> j -> k (standard naive matrix multiply)
void matmul_sequential(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols != B.rows");
    }

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    if (C.rows != M || C.cols != N) {
        C = Matrix(M, N, 0.0f);
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// Phase 2: OpenMP Parallelized Naive GEMM
void matmul_openmp(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols != B.rows");
    }

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    if (C.rows != M || C.cols != N) {
        C = Matrix(M, N, 0.0f);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// Phase 2: Cache-Friendly Loop Reordered GEMM (i-k-j traversal)
// Why this matters: B is accessed row-by-row (unit-stride memory access),
// yielding massive L1/L2 cache hit rate improvements over naive i-j-k.
void matmul_cache_friendly(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols != B.rows");
    }

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    C = Matrix(M, N, 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float r = A(i, k);
            for (int j = 0; j < N; ++j) {
                C(i, j) += r * B(k, j);
            }
        }
    }
}

// Phase 2: OpenMP + Cache-Blocked (Tiled) GEMM
void matmul_openmp_blocked(const Matrix& A, const Matrix& B, Matrix& C, int blockSize) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols != B.rows");
    }

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    C = Matrix(M, N, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < M; bi += blockSize) {
        for (int bj = 0; bj < N; bj += blockSize) {
            for (int bk = 0; bk < K; bk += blockSize) {
                int i_end = std::min(bi + blockSize, M);
                int j_end = std::min(bj + blockSize, N);
                int k_end = std::min(bk + blockSize, K);

                for (int i = bi; i < i_end; ++i) {
                    for (int k = bk; k < k_end; ++k) {
                        float r = A(i, k);
                        for (int j = bj; j < j_end; ++j) {
                            C(i, j) += r * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

} // namespace hpc
