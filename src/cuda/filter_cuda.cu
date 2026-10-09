#include "cuda/filter_cuda.cuh"
#include "cuda/cuda_utils.cuh"
#include <algorithm>

namespace hpc {

// ----------------------------------------------------------------------------
// Phase 3 Kernel: Naive 2D Convolution using Global Memory
// Each thread maps to an (x, y) coordinate in the output image.
// ----------------------------------------------------------------------------
__global__ void applyFilterNaiveKernel(const float* __restrict__ d_input,
                                       float* __restrict__ d_output,
                                       int width, int height,
                                       const float* __restrict__ d_kernel,
                                       int kSize, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        int py = max(0, min(y + ky, height - 1));
        int kRow = ky + radius;

        for (int kx = -radius; kx <= radius; ++kx) {
            int px = max(0, min(x + kx, width - 1));
            int kCol = kx + radius;

            sum += d_input[py * width + px] * d_kernel[kRow * kSize + kCol];
        }
    }

    d_output[y * width + x] = fminf(fmaxf(sum, 0.0f), 255.0f);
}

// ----------------------------------------------------------------------------
// Phase 3 Kernel: Naive Sobel Edge Detection using Global Memory
// ----------------------------------------------------------------------------
__global__ void applySobelNaiveKernel(const float* __restrict__ d_input,
                                      float* __restrict__ d_output,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sobel horizontal (kx) and vertical (ky) kernels
    const float kx[3][3] = {
        {-1.0f, 0.0f, 1.0f},
        {-2.0f, 0.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f}
    };
    const float ky[3][3] = {
        {-1.0f, -2.0f, -1.0f},
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  1.0f}
    };

    float gx = 0.0f;
    float gy = 0.0f;

    for (int r = -1; r <= 1; ++r) {
        int py = max(0, min(y + r, height - 1));
        for (int c = -1; c <= 1; ++c) {
            int px = max(0, min(x + c, width - 1));
            float p = d_input[py * width + px];
            gx += p * kx[r + 1][c + 1];
            gy += p * ky[r + 1][c + 1];
        }
    }

    float mag = sqrtf(gx * gx + gy * gy);
    d_output[y * width + x] = fminf(fmaxf(mag, 0.0f), 255.0f);
}

// ----------------------------------------------------------------------------
// Host Wrapper: Naive 2D Convolution (cudaMalloc, cudaMemcpy, Grid launch)
// ----------------------------------------------------------------------------
void filter_naive_cuda(const Image& h_input, Image& h_output, const Kernel& kernel, float* out_kernel_ms) {
    const int width = h_input.width;
    const int height = h_input.height;
    const size_t imgBytes = width * height * sizeof(float);
    const size_t kernelBytes = kernel.weights.size() * sizeof(float);

    if (h_output.width != width || h_output.height != height) {
        h_output = Image(width, height, 1);
    }

    // 1. Host-Device Memory Allocation
    float *d_input = nullptr, *d_output = nullptr, *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));

    // 2. Transfer input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.raw(), imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.weights.data(), kernelBytes, cudaMemcpyHostToDevice));

    // 3. Configure 2D Grid / Block Geometry
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 4. Execute Kernel with CUDA Event Timing
    CudaTimer timer;
    timer.start();
    applyFilterNaiveKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, d_kernel, kernel.size, kernel.radius);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    // 5. Transfer result back to host
    CUDA_CHECK(cudaMemcpy(h_output.raw(), d_output, imgBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
}

// ----------------------------------------------------------------------------
// Host Wrapper: Naive Sobel (cudaMalloc, cudaMemcpy, Grid launch)
// ----------------------------------------------------------------------------
void sobel_naive_cuda(const Image& h_input, Image& h_output, float* out_kernel_ms) {
    const int width = h_input.width;
    const int height = h_input.height;
    const size_t imgBytes = width * height * sizeof(float);

    if (h_output.width != width || h_output.height != height) {
        h_output = Image(width, height, 1);
    }

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.raw(), imgBytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    CudaTimer timer;
    timer.start();
    applySobelNaiveKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_output.raw(), d_output, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// ----------------------------------------------------------------------------
// Phase 4 Kernel: Tiled 2D Convolution with Shared Memory Halo Management
// ----------------------------------------------------------------------------
__global__ void applyFilterTiledKernel(const float* __restrict__ d_input,
                                       float* __restrict__ d_output,
                                       int width, int height,
                                       const float* __restrict__ d_kernel,
                                       int kSize, int radius) {
    extern __shared__ float s_tile[];

    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;
    const int tileTotal = tileW * tileH;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    // 1. Cooperative Loading of interior tile + outer halo cells
    for (int i = tid; i < tileTotal; i += numThreads) {
        int sy = i / tileW;
        int sx = i % tileW;

        int gy = blockIdx.y * blockDim.y + sy - radius;
        int gx = blockIdx.x * blockDim.x + sx - radius;

        // Clamp to edge
        gy = max(0, min(gy, height - 1));
        gx = max(0, min(gx, width - 1));

        s_tile[sy * tileW + sx] = d_input[gy * width + gx];
    }

    // 2. Barrier Synchronization: Ensure full tile + halos are staged
    __syncthreads();

    // 3. Compute convolution directly from on-chip shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int sy = threadIdx.y + radius;
        int sx = threadIdx.x + radius;

        for (int ky = -radius; ky <= radius; ++ky) {
            int kRow = ky + radius;
            for (int kx = -radius; kx <= radius; ++kx) {
                int kCol = kx + radius;
                sum += s_tile[(sy + ky) * tileW + (sx + kx)] * d_kernel[kRow * kSize + kCol];
            }
        }
        d_output[y * width + x] = fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// ----------------------------------------------------------------------------
// Phase 4 Kernel: Tiled Sobel Edge Detection with Shared Memory Halo Management
// ----------------------------------------------------------------------------
__global__ void applySobelTiledKernel(const float* __restrict__ d_input,
                                      float* __restrict__ d_output,
                                      int width, int height) {
    extern __shared__ float s_tile[];

    const int radius = 1;
    const int tileW = blockDim.x + 2 * radius;
    const int tileH = blockDim.y + 2 * radius;
    const int tileTotal = tileW * tileH;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    for (int i = tid; i < tileTotal; i += numThreads) {
        int sy = i / tileW;
        int sx = i % tileW;

        int gy = blockIdx.y * blockDim.y + sy - radius;
        int gx = blockIdx.x * blockDim.x + sx - radius;

        gy = max(0, min(gy, height - 1));
        gx = max(0, min(gx, width - 1));

        s_tile[sy * tileW + sx] = d_input[gy * width + gx];
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sy = threadIdx.y + radius;
        int sx = threadIdx.x + radius;

        float p00 = s_tile[(sy - 1) * tileW + (sx - 1)];
        float p01 = s_tile[(sy - 1) * tileW + sx];
        float p02 = s_tile[(sy - 1) * tileW + (sx + 1)];

        float p10 = s_tile[sy * tileW + (sx - 1)];
        float p12 = s_tile[sy * tileW + (sx + 1)];

        float p20 = s_tile[(sy + 1) * tileW + (sx - 1)];
        float p21 = s_tile[(sy + 1) * tileW + sx];
        float p22 = s_tile[(sy + 1) * tileW + (sx + 1)];

        float gx_val = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
        float gy_val = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

        float mag = sqrtf(gx_val * gx_val + gy_val * gy_val);
        d_output[y * width + x] = fminf(fmaxf(mag, 0.0f), 255.0f);
    }
}

// ----------------------------------------------------------------------------
// Host Wrapper: Tiled 2D Convolution with Shared Memory
// ----------------------------------------------------------------------------
void filter_tiled_cuda(const Image& h_input, Image& h_output, const Kernel& kernel, float* out_kernel_ms) {
    const int width = h_input.width;
    const int height = h_input.height;
    const size_t imgBytes = width * height * sizeof(float);
    const size_t kernelBytes = kernel.weights.size() * sizeof(float);

    if (h_output.width != width || h_output.height != height) {
        h_output = Image(width, height, 1);
    }

    float *d_input = nullptr, *d_output = nullptr, *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.raw(), imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.weights.data(), kernelBytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    const int radius = kernel.radius;
    const size_t sharedMemBytes = (blockDim.x + 2 * radius) * (blockDim.y + 2 * radius) * sizeof(float);

    CudaTimer timer;
    timer.start();
    applyFilterTiledKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_input, d_output, width, height, d_kernel, kernel.size, radius);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_output.raw(), d_output, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
}

// ----------------------------------------------------------------------------
// Host Wrapper: Tiled Sobel with Shared Memory
// ----------------------------------------------------------------------------
void sobel_tiled_cuda(const Image& h_input, Image& h_output, float* out_kernel_ms) {
    const int width = h_input.width;
    const int height = h_input.height;
    const size_t imgBytes = width * height * sizeof(float);

    if (h_output.width != width || h_output.height != height) {
        h_output = Image(width, height, 1);
    }

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_output, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.raw(), imgBytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    const size_t sharedMemBytes = (blockDim.x + 2) * (blockDim.y + 2) * sizeof(float);

    CudaTimer timer;
    timer.start();
    applySobelTiledKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_input, d_output, width, height);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    if (out_kernel_ms) {
        *out_kernel_ms = timer.elapsed_ms();
    }

    CUDA_CHECK(cudaMemcpy(h_output.raw(), d_output, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

} // namespace hpc
