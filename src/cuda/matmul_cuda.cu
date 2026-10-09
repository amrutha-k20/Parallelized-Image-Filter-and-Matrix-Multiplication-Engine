#include "cuda/matmul_cuda.cuh"
#include "cuda/cuda_utils.cuh"
#include <stdexcept>

namespace hpc {

// ----------------------------------------------------------------------------
// Phase 3 Kernel: Naive GEMM using Global Memory
// Each thread computes one element C(row, col)
// ----------------------------------------------------------------------------
__global__ void matmulNaiveKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Host Wrapper: Naive CUDA GEMM
// ----------------------------------------------------------------------------
void matmul_naive_cuda(const Matrix& h_A, const Matrix& h_B, Matrix& h_C, float* out_kernel_ms) {
    if (h_A.cols != h_B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch in CUDA GEMM");
    }

    const int M = h_A.rows;
    const int K = h_A.cols;
    const int N = h_B.cols;

    if (h_C.rows != M || h_C.cols != N) {
        h_C = Matrix(M, N, 0.0f);
    }

    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.raw(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.raw(), bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    CudaTimer timer;
    timer.start();
    matmulNaiveKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_C.raw(), d_C, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ----------------------------------------------------------------------------
// Phase 4 Kernel: Tiled GEMM using On-Chip Shared Memory
// ----------------------------------------------------------------------------
template <int TILE_DIM>
__global__ void matmulTiledKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    float sum = 0.0f;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_DIM + tx;
        if (row < M && aCol < K) {
            s_A[ty][tx] = A[row * K + aCol];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        int bRow = t * TILE_DIM + ty;
        if (bRow < K && col < N) {
            s_B[ty][tx] = B[bRow * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Host Wrapper: Tiled CUDA GEMM
// ----------------------------------------------------------------------------
void matmul_tiled_cuda(const Matrix& h_A, const Matrix& h_B, Matrix& h_C, float* out_kernel_ms) {
    if (h_A.cols != h_B.rows) {
        throw std::invalid_argument("Matrix dimension mismatch in CUDA GEMM");
    }

    const int M = h_A.rows;
    const int K = h_A.cols;
    const int N = h_B.cols;

    if (h_C.rows != M || h_C.cols != N) {
        h_C = Matrix(M, N, 0.0f);
    }

    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.raw(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.raw(), bytesB, cudaMemcpyHostToDevice));

    constexpr int TILE_DIM = 16;
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
                 (M + TILE_DIM - 1) / TILE_DIM);

    CudaTimer timer;
    timer.start();
    matmulTiledKernel<TILE_DIM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_C.raw(), d_C, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

} // namespace hpc
