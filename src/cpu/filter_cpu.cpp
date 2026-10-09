#include "cpu/filter_cpu.hpp"
#include <algorithm>
#include <cmath>

namespace hpc {

Kernel createGaussianBlur3x3() {
    // 3x3 Gaussian kernel with sigma ~= 1.0 (sum = 16)
    return Kernel(3, {
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
        2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
        1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
    });
}

Kernel createGaussianBlur5x5() {
    // 5x5 Gaussian kernel with sigma ~= 1.4 (sum = 256)
    return Kernel(5, {
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        6.0f / 256.0f, 24.0f / 256.0f, 36.0f / 256.0f, 24.0f / 256.0f, 6.0f / 256.0f,
        4.0f / 256.0f, 16.0f / 256.0f, 24.0f / 256.0f, 16.0f / 256.0f, 4.0f / 256.0f,
        1.0f / 256.0f,  4.0f / 256.0f,  6.0f / 256.0f,  4.0f / 256.0f, 1.0f / 256.0f
    });
}

Kernel createSobelX() {
    // 3x3 Sobel Horizontal Gradient
    return Kernel(3, {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    });
}

Kernel createSobelY() {
    // 3x3 Sobel Vertical Gradient
    return Kernel(3, {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    });
}

// Sequential 2D convolution with clamp-to-edge boundary handling
void convolve2D_sequential(const Image& input, Image& output, const Kernel& kernel) {
    const int width = input.width;
    const int height = input.height;
    const int radius = kernel.radius;
    const int kSize = kernel.size;

    if (output.width != width || output.height != height) {
        output = Image(width, height, 1);
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ++ky) {
                int py = std::min(std::max(y + ky, 0), height - 1);
                int kRow = ky + radius;

                for (int kx = -radius; kx <= radius; ++kx) {
                    int px = std::min(std::max(x + kx, 0), width - 1);
                    int kCol = kx + radius;

                    sum += input.at(py, px) * kernel.weights[kRow * kSize + kCol];
                }
            }

            output.at(y, x) = std::min(std::max(sum, 0.0f), 255.0f);
        }
    }
}

// Sequential Sobel Edge Detection (computes gradient magnitude sqrt(Gx^2 + Gy^2))
void applySobel_sequential(const Image& input, Image& output) {
    const int width = input.width;
    const int height = input.height;

    if (output.width != width || output.height != height) {
        output = Image(width, height, 1);
    }

    // Sobel kernels
    const float kx_w[3][3] = {
        {-1.0f, 0.0f, 1.0f},
        {-2.0f, 0.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f}
    };
    const float ky_w[3][3] = {
        {-1.0f, -2.0f, -1.0f},
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  1.0f}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                int py = std::min(std::max(y + ky, 0), height - 1);
                int r = ky + 1;

                for (int kx = -1; kx <= 1; ++kx) {
                    int px = std::min(std::max(x + kx, 0), width - 1);
                    int c = kx + 1;

                    float p = input.at(py, px);
                    gx += p * kx_w[r][c];
                    gy += p * ky_w[r][c];
                }
            }

            float mag = std::sqrt(gx * gx + gy * gy);
            output.at(y, x) = std::min(std::max(mag, 0.0f), 255.0f);
        }
    }
}

// Phase 2: OpenMP Parallel 2D Convolution with #pragma omp parallel for collapse(2)
void convolve2D_openmp(const Image& input, Image& output, const Kernel& kernel, ScheduleType sched, int chunkSize) {
    const int width = input.width;
    const int height = input.height;
    const int radius = kernel.radius;
    const int kSize = kernel.size;

    if (output.width != width || output.height != height) {
        output = Image(width, height, 1);
    }

    if (sched == ScheduleType::STATIC) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky) {
                    int py = std::min(std::max(y + ky, 0), height - 1);
                    int kRow = ky + radius;
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int kCol = kx + radius;
                        sum += input.at(py, px) * kernel.weights[kRow * kSize + kCol];
                    }
                }
                output.at(y, x) = std::min(std::max(sum, 0.0f), 255.0f);
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(dynamic, chunkSize)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ++ky) {
                    int py = std::min(std::max(y + ky, 0), height - 1);
                    int kRow = ky + radius;
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int kCol = kx + radius;
                        sum += input.at(py, px) * kernel.weights[kRow * kSize + kCol];
                    }
                }
                output.at(y, x) = std::min(std::max(sum, 0.0f), 255.0f);
            }
        }
    }
}

// Phase 2: OpenMP Parallel Sobel Edge Detection
void applySobel_openmp(const Image& input, Image& output, ScheduleType sched, int chunkSize) {
    const int width = input.width;
    const int height = input.height;

    if (output.width != width || output.height != height) {
        output = Image(width, height, 1);
    }

    const float kx_w[3][3] = {
        {-1.0f, 0.0f, 1.0f},
        {-2.0f, 0.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f}
    };
    const float ky_w[3][3] = {
        {-1.0f, -2.0f, -1.0f},
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  1.0f}
    };

    if (sched == ScheduleType::STATIC) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float gx = 0.0f;
                float gy = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int py = std::min(std::max(y + ky, 0), height - 1);
                    int r = ky + 1;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int c = kx + 1;
                        float p = input.at(py, px);
                        gx += p * kx_w[r][c];
                        gy += p * ky_w[r][c];
                    }
                }
                float mag = std::sqrt(gx * gx + gy * gy);
                output.at(y, x) = std::min(std::max(mag, 0.0f), 255.0f);
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(dynamic, chunkSize)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float gx = 0.0f;
                float gy = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int py = std::min(std::max(y + ky, 0), height - 1);
                    int r = ky + 1;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = std::min(std::max(x + kx, 0), width - 1);
                        int c = kx + 1;
                        float p = input.at(py, px);
                        gx += p * kx_w[r][c];
                        gy += p * ky_w[r][c];
                    }
                }
                float mag = std::sqrt(gx * gx + gy * gy);
                output.at(y, x) = std::min(std::max(mag, 0.0f), 255.0f);
            }
        }
    }
}

} // namespace hpc
