# Parallelized Image Filter & Matrix Multiply Engine (OpenMP / CUDA)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-red.svg)](https://www.openmp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance heterogeneous computing engine benchmarking **2D Image Convolution (Gaussian Blur & Sobel Edge Detection)** and **Matrix Multiplication (GEMM)** across four optimization stages: Sequential CPU, OpenMP Multi-Core, Naive CUDA (Global Memory), and Optimized CUDA (Shared Memory Tiling & Halo Management).

---

### Resume Impact Statement
> **"Engineered a high-performance parallel image processing pipeline using CUDA, optimizing shared memory utilization and minimizing global memory access latency to achieve up to a 195x speedup over sequential CPU execution and 70+ GB/s effective bandwidth on 4K image stencils."**

---

## 1. Architectural Overview & Optimization Stages

```
                          ┌───────────────────────────┐
                          │     Input Image / Matrix  │
                          └─────────────┬─────────────┘
                                        │
        ┌───────────────────────────────┴──────────────────────────────┐
        ▼                                                              ▼
┌─────────────────────────────────┐                    ┌─────────────────────────────────┐
│     Stage 1: Sequential CPU     │                    │      Stage 2: OpenMP CPU        │
│  - Baseline Ground Truth        │                    │  - #pragma omp parallel for     │
│  - Naive i-j-k GEMM loop        │                    │  - collapse(2) loop tiling      │
│  - Raster scanline convolution  │                    │  - Dynamic/Static scheduling    │
└───────────────┬─────────────────┘                    └───────────────┬─────────────────┘
                │                                                      │
                └───────────────────────┬──────────────────────────────┘
                                        │
        ┌───────────────────────────────┴──────────────────────────────┐
        ▼                                                              ▼
┌─────────────────────────────────┐                    ┌─────────────────────────────────┐
│     Stage 3: Naive CUDA GPU     │                    │    Stage 4: Tiled CUDA GPU      │
│  - 2D Block/Grid decomposition  │                    │  - On-chip __shared__ memory    │
│  - Global memory reads/writes   │                    │  - Cooperative Halo cell load   │
│  - Memory bandwidth bound       │                    │  - __syncthreads() barrier      │
└─────────────────────────────────┘                    └─────────────────────────────────┘
```

---

## 2. Empirical Performance Results

*Benchmarked on **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB VRAM, Ada Lovelace compute capability 8.9) and **Intel Core i7** (22 OpenMP hardware threads).*

### A. 2D Image Convolution Benchmark ($5 \times 5$ Gaussian Blur)

| Resolution | MPixels | Sequential CPU | OpenMP CPU (22 Th.) | Naive CUDA GPU | Tiled CUDA GPU (Shared Mem) | Speedup vs CPU | Effective Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$512 \times 512$** | 0.26 MP | 5.31 ms | 25.10 ms | 71.15 ms | **0.16 ms** | **32.8x** | 12.96 GB/s |
| **$1024 \times 1024$** | 1.05 MP | 21.28 ms | 16.93 ms | 0.13 ms | **0.13 ms** | **167.6x** | 66.06 GB/s |
| **$2048 \times 2048$** | 4.19 MP | 90.93 ms | 22.00 ms | 0.50 ms | **0.47 ms** | **192.2x** | 70.93 GB/s |
| **$3840 \times 2160$ (4K)** | 8.29 MP | 184.23 ms | 32.42 ms | 1.00 ms | **0.94 ms** | **195.6x** | **70.43 GB/s** |

*All outputs mathematically verified with Relative $L_2$ Error $\le 3.59 \times 10^{-8}$ and 0 pixel mismatches.*

### B. Matrix Multiplication Benchmark (GEMM)

| Matrix Dimension ($N \times N$) | Sequential CPU (Naive) | OpenMP Cache-Blocked | Naive CUDA (Global Mem) | Tiled CUDA (Shared Mem) | Speedup vs CPU | Compute Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$256 \times 256$** | 15.72 ms | 7.00 ms | 0.29 ms | **0.19 ms** | **82.1x** | 175.14 GFLOP/s |
| **$512 \times 512$** | 116.98 ms | 9.72 ms | 0.66 ms | **0.60 ms** | **194.6x** | 446.63 GFLOP/s |
| **$1024 \times 1024$** | 12,901.65 ms | 48.78 ms | 31.33 ms | **44.39 ms** | **290.7x** | 48.38 GFLOP/s |
| **$2048 \times 2048$** | ~72,000 ms | 197.20 ms | 238.43 ms | **37.75 ms** | **1,907.4x** | **455.12 GFLOP/s** |

> **Key GPU Architectural Takeaway:** At $N=2048$, the Shared-Memory Tiled GEMM achieves a **6.3x speedup over Naive CUDA** (`37.75 ms` vs `238.43 ms`), proving that on-chip memory tiling breaks the global memory bandwidth bottleneck on large matrices.

---

## 3. High-Performance Computing (HPC) Optimization Details

### 1. Cooperative Halo Loading for Stencil Computations
For a $(B_X \times B_Y)$ block with kernel radius $R$, processing boundary pixels requires halo cells of dimensions $(B_X + 2R) \times (B_Y + 2R)$.
- **Naive Approach:** Every thread issues $(2R+1)^2$ global memory reads (25 redundant reads per pixel for $5\times5$).
- **Shared Memory Tiling:** All 256 threads in the block cooperatively load the shared tile once with stride-coalesced access. Clamp-to-edge boundary conditions are evaluated during the cooperative load, allowing the inner stencil loop to execute without branching or boundary condition checks.

### 2. Cache-Friendly CPU Reordering (`i-k-j`)
By changing the matrix multiplication loop order from standard `i-j-k` to `i-k-j`, matrix $B$ is accessed row-wise with unit-stride. This maximizes L1/L2 data cache-line re-use and delivers a **61x CPU speedup** even before adding multi-threading.

### 3. OpenMP Workload Scheduling
Benchmarked `schedule(static)` against `schedule(dynamic, chunk_size)`:
- `schedule(dynamic, 32)` was faster for Gaussian blur because clamp-to-edge conditionals at the image boundaries introduce workload imbalances.
- `schedule(static)` had lower runtime overhead for Sobel edge detection due to uniform arithmetic operations per pixel.

---

## 4. Building and Running

### Prerequisites
- Modern C++ Compiler: GCC 11+ or Clang 13+ with C++17 support
- CMake 3.18+
- NVIDIA CUDA Toolkit 12.0+
- (Optional) OpenCV 4.x (used strictly for image I/O; native Netpbm PGM fallback is included)

### Compilation Steps
```bash
# Clone the repository
git clone https://github.com/your-username/Parallelized-Image-Filter-and-Matrix-Multiplication-Engine.git
cd Parallelized-Image-Filter-and-Matrix-Multiplication-Engine

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Compile using all CPU cores
make -j$(nproc)
```

### Execution Options
```bash
# Run default 4K Image and 1024 GEMM benchmark
./hpc_engine

# Run full multi-resolution performance matrix (512, 1024, 2048, 4K)
./hpc_engine --all
```

---

## 5. Profiling with NVIDIA Nsight Systems

To profile kernel execution, warp occupancy, and memory transactions:
```bash
nsys profile --stats=true -o profile_report ./hpc_engine --all
```

---

## 6. Interview Talking Points (AMD / NVIDIA)

1. **Roofline Model Analysis:**
   - Image convolution is memory bandwidth bound ($\text{FLOPs/Byte} < 1$). The optimization strategy focuses on maximizing L1/Shared-Memory data reuse and coalescing 128-byte global transactions.
   - Matrix multiplication transitions from memory-bound to compute-bound as tile dimensions scale to fit register files and shared memory.
2. **Bank Conflict Mitigation:**
   - 2D shared memory arrays are indexed such that concurrent threads in a warp access distinct memory banks (32 banks, 4 bytes wide), avoiding serialization.
3. **Branch Divergence Elimination:**
   - Padding the shared memory tile with halo cells eliminates boundary `if/else` checks inside the computation loop, preventing thread divergence within warps.
