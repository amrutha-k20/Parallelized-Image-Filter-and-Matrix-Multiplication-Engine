#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " (" << cudaGetErrorString(err) << ") \"" \
                      << #call << "\"\n"; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

namespace hpc {

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&m_start));
        CUDA_CHECK(cudaEventCreate(&m_stop));
    }

    ~CudaTimer() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(m_start, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(m_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(m_stop));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }

private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};

inline void printGpuInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU detected or CUDA error: "
                  << cudaGetErrorString(err) << "\n";
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[GPU Info] Device: " << prop.name << "\n";
    std::cout << "[GPU Info] Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "[GPU Info] Total Global Memory: "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "[GPU Info] Shared Memory per Block: "
              << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << "[GPU Info] Warp Size: " << prop.warpSize << " threads\n";
}

} // namespace hpc
