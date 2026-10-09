#pragma once

#include "common/types.hpp"

namespace hpc {

// Filter Kernels Factory
Kernel createGaussianBlur3x3();
Kernel createGaussianBlur5x5();
Kernel createSobelX();
Kernel createSobelY();

// Phase 1: Sequential CPU Implementation
void convolve2D_sequential(const Image& input, Image& output, const Kernel& kernel);
void applySobel_sequential(const Image& input, Image& output);

// Phase 2: OpenMP Parallel CPU Implementation
enum class ScheduleType {
    STATIC,
    DYNAMIC
};

void convolve2D_openmp(const Image& input, Image& output, const Kernel& kernel, ScheduleType sched = ScheduleType::STATIC, int chunkSize = 32);
void applySobel_openmp(const Image& input, Image& output, ScheduleType sched = ScheduleType::STATIC, int chunkSize = 32);

} // namespace hpc
