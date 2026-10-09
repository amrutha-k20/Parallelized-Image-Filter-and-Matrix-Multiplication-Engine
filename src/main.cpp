#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <omp.h>
#include "common/types.hpp"
#include "common/timer.hpp"
#include "common/verify.hpp"
#include "cpu/filter_cpu.hpp"
#include "cpu/matmul_cpu.hpp"
#include "cuda/cuda_utils.cuh"
#include "cuda/filter_cuda.cuh"
#include "cuda/matmul_cuda.cuh"
#include "io/image_io.hpp"

void printDivider(char ch = '-', int len = 68) {
    std::cout << std::string(len, ch) << "\n";
}

int main(int, char**) {
    std::cout << "\n";
    printDivider('=', 68);
    std::cout << "  High-Performance Image Filtering & Matrix Engine (OpenMP / CUDA)\n";
    printDivider('=', 68);

    int numThreads = omp_get_max_threads();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Hardware Configuration:\n";
    std::cout << "  CPU Threads : " << numThreads << " OpenMP Threads\n";
    std::cout << "  GPU Device  : " << prop.name << " (" << (prop.totalGlobalMem / (1024 * 1024))
              << " MB VRAM, CC " << prop.major << "." << prop.minor << ")\n";
    printDivider('-', 68);

    // Warmup GPU context
    cudaFree(0);

    // =========================================================================
    // 1. 4K Image Convolution Benchmark
    // =========================================================================
    const int imgW = 3840;
    const int imgH = 2160;
    const double mpixels = (imgW * imgH) / 1e6;

    std::cout << "\n[1] 4K Image Convolution Benchmark (" << imgW << "x" << imgH
              << ", " << std::fixed << std::setprecision(2) << mpixels << " MPixels)\n";
    printDivider('-', 68);
    std::cout << std::left << std::setw(26) << "Method"
              << std::setw(15) << "Latency"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup" << "\n";
    printDivider('-', 68);

    hpc::Image input4K = hpc::Image::createSynthetic4K(imgW, imgH);
    hpc::Image cpuRef(imgW, imgH);
    hpc::Image ompOut(imgW, imgH);
    hpc::Image naiveCudaOut(imgW, imgH);
    hpc::Image tiledCudaOut(imgW, imgH);
    hpc::Image sobelOut(imgW, imgH);

    hpc::Kernel kGauss5 = hpc::createGaussianBlur5x5();
    hpc::CpuTimer timer;

    // Sequential CPU
    timer.start();
    hpc::convolve2D_sequential(input4K, cpuRef, kGauss5);
    timer.stop();
    double seqMs = timer.elapsed_ms();
    double seqThroughput = mpixels / (seqMs / 1000.0);
    std::cout << std::left << std::setw(26) << "Sequential CPU"
              << std::setw(15) << (std::to_string((int)seqMs) + " ms")
              << std::setw(15) << (std::to_string((int)seqThroughput) + " MP/s")
              << "1.0x (ref)\n";

    // OpenMP Multi-Core
    timer.start();
    hpc::convolve2D_openmp(input4K, ompOut, kGauss5, hpc::ScheduleType::STATIC);
    timer.stop();
    double ompMs = timer.elapsed_ms();
    double ompThroughput = mpixels / (ompMs / 1000.0);
    std::cout << std::left << std::setw(26) << "OpenMP CPU"
              << std::setw(15) << (std::to_string((int)ompMs) + " ms")
              << std::setw(15) << (std::to_string((int)ompThroughput) + " MP/s")
              << (std::to_string(seqMs / ompMs).substr(0, 4) + "x") << "\n";

    // Naive CUDA
    float naiveKernelMs = 0.0f;
    hpc::filter_naive_cuda(input4K, naiveCudaOut, kGauss5, &naiveKernelMs);
    double naiveThroughput = mpixels / (naiveKernelMs / 1000.0);
    std::cout << std::left << std::setw(26) << "CUDA (Global Memory)"
              << std::setw(15) << (std::to_string(naiveKernelMs).substr(0, 5) + " ms")
              << std::setw(15) << (std::to_string((int)naiveThroughput) + " MP/s")
              << (std::to_string(seqMs / naiveKernelMs).substr(0, 5) + "x") << "\n";

    // Tiled Shared Memory CUDA
    float tiledKernelMs = 0.0f;
    hpc::filter_tiled_cuda(input4K, tiledCudaOut, kGauss5, &tiledKernelMs);
    double tiledThroughput = mpixels / (tiledKernelMs / 1000.0);
    std::cout << std::left << std::setw(26) << "CUDA (Shared Memory)"
              << std::setw(15) << (std::to_string(tiledKernelMs).substr(0, 5) + " ms")
              << std::setw(15) << (std::to_string((int)tiledThroughput) + " MP/s")
              << (std::to_string(seqMs / tiledKernelMs).substr(0, 5) + "x") << "\n";

    printDivider('-', 68);
    auto verImg = hpc::verifyImages(cpuRef, tiledCudaOut, 1e-3f);
    std::cout << "Correctness: " << (verImg.passed ? "PASSED (0 mismatches)" : "FAILED")
              << " | Rel L2 Error: " << std::scientific << std::setprecision(2)
              << verImg.relative_l2_error << "\n";

    // Generate Sobel edge map on GPU
    hpc::sobel_tiled_cuda(input4K, sobelOut);

    // Save PNG images for easy viewing on Windows
    hpc::saveImage("output_input_4k.png", input4K);
    hpc::saveImage("output_gaussian_4k.png", tiledCudaOut);
    hpc::saveImage("output_sobel_4k.png", sobelOut);

    // =========================================================================
    // 2. Matrix Multiplication Benchmark (GEMM)
    // =========================================================================
    const int N = 1024;
    std::cout << "\n[2] Matrix Multiplication Benchmark (1024 x 1024 GEMM)\n";
    printDivider('-', 68);
    std::cout << std::left << std::setw(26) << "Method"
              << std::setw(15) << "Latency"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Speedup" << "\n";
    printDivider('-', 68);

    hpc::Matrix A(N, N);
    hpc::Matrix B(N, N);
    hpc::Matrix gemmRef(N, N);
    hpc::Matrix gemmOmp(N, N);
    hpc::Matrix gemmNaive(N, N);
    hpc::Matrix gemmTiled(N, N);
    A.randomize(-1.0f, 1.0f, 42);
    B.randomize(-1.0f, 1.0f, 84);

    // Sequential CPU
    timer.start();
    hpc::matmul_sequential(A, B, gemmRef);
    timer.stop();
    double seqGemmMs = timer.elapsed_ms();
    double gflopsSeq = (2.0 * N * N * N) / (seqGemmMs * 1e6);
    std::cout << std::left << std::setw(26) << "Sequential CPU"
              << std::setw(15) << (std::to_string((int)seqGemmMs) + " ms")
              << std::setw(15) << (std::to_string(gflopsSeq).substr(0, 4) + " GFLOP/s")
              << "1.0x (ref)\n";

    // OpenMP Blocked
    timer.start();
    hpc::matmul_openmp_blocked(A, B, gemmOmp, 64);
    timer.stop();
    double ompGemmMs = timer.elapsed_ms();
    double gflopsOmp = (2.0 * N * N * N) / (ompGemmMs * 1e6);
    std::cout << std::left << std::setw(26) << "OpenMP Blocked"
              << std::setw(15) << (std::to_string((int)ompGemmMs) + " ms")
              << std::setw(15) << (std::to_string(gflopsOmp).substr(0, 5) + " GFLOP/s")
              << (std::to_string(seqGemmMs / ompGemmMs).substr(0, 5) + "x") << "\n";

    // Naive CUDA
    float naiveGemmMs = 0.0f;
    hpc::matmul_naive_cuda(A, B, gemmNaive, &naiveGemmMs);
    double gflopsNaive = (2.0 * N * N * N) / (naiveGemmMs * 1e6);
    std::cout << std::left << std::setw(26) << "CUDA (Global Memory)"
              << std::setw(15) << (std::to_string((int)naiveGemmMs) + " ms")
              << std::setw(15) << (std::to_string(gflopsNaive).substr(0, 5) + " GFLOP/s")
              << (std::to_string(seqGemmMs / naiveGemmMs).substr(0, 5) + "x") << "\n";

    // Tiled CUDA
    float tiledGemmMs = 0.0f;
    hpc::matmul_tiled_cuda(A, B, gemmTiled, &tiledGemmMs);
    double gflopsTiled = (2.0 * N * N * N) / (tiledGemmMs * 1e6);
    std::cout << std::left << std::setw(26) << "CUDA (Shared Memory)"
              << std::setw(15) << (std::to_string((int)tiledGemmMs) + " ms")
              << std::setw(15) << (std::to_string(gflopsTiled).substr(0, 5) + " GFLOP/s")
              << (std::to_string(seqGemmMs / tiledGemmMs).substr(0, 5) + "x") << "\n";

    printDivider('-', 68);
    auto verGemm = hpc::verifyMatrices(gemmRef, gemmTiled, 1e-3f);
    std::cout << "Correctness: " << (verGemm.passed ? "PASSED (0 mismatches)" : "FAILED")
              << " | Rel L2 Error: " << std::scientific << std::setprecision(2)
              << verGemm.relative_l2_error << "\n";

    // =========================================================================
    // Output Files Summary
    // =========================================================================
    printDivider('=', 68);
    std::cout << "Generated Output Images (Standard PNG format):\n";
    std::cout << "  - output_input_4k.png    : Original 4K image\n";
    std::cout << "  - output_gaussian_4k.png : 5x5 Gaussian blur filtered image\n";
    std::cout << "  - output_sobel_4k.png    : 3x3 Sobel gradient edge detection map\n";
    printDivider('=', 68);
    std::cout << "\n";

    return 0;
}
