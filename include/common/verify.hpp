#pragma once

#include "types.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hpc {

struct VerificationResult {
    bool passed = false;
    float max_abs_diff = 0.0f;
    float rmse = 0.0f;
    float relative_l2_error = 0.0f;
    int mismatched_elements = 0;
};

inline VerificationResult verifyBuffers(const float* ref, const float* test, size_t count, float tolerance = 1e-4f) {
    VerificationResult res;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    float max_diff = 0.0f;
    int mismatches = 0;

    for (size_t i = 0; i < count; ++i) {
        float r = ref[i];
        float t = test[i];
        float diff = std::abs(r - t);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > tolerance) {
            mismatches++;
        }
        sum_sq_diff += diff * diff;
        sum_sq_ref += r * r;
    }

    res.max_abs_diff = max_diff;
    res.rmse = static_cast<float>(std::sqrt(sum_sq_diff / count));
    res.relative_l2_error = static_cast<float>(std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-9));
    res.mismatched_elements = mismatches;
    res.passed = (mismatches == 0) || (res.relative_l2_error < tolerance);

    return res;
}

inline VerificationResult verifyImages(const Image& ref, const Image& test, float tolerance = 1e-4f) {
    if (ref.width != test.width || ref.height != test.height || ref.channels != test.channels) {
        std::cerr << "Dimension mismatch in image verification!\n";
        VerificationResult res;
        res.passed = false;
        return res;
    }
    return verifyBuffers(ref.raw(), test.raw(), ref.size(), tolerance);
}

inline VerificationResult verifyMatrices(const Matrix& ref, const Matrix& test, float tolerance = 1e-4f) {
    if (ref.rows != test.rows || ref.cols != test.cols) {
        std::cerr << "Dimension mismatch in matrix verification!\n";
        VerificationResult res;
        res.passed = false;
        return res;
    }
    return verifyBuffers(ref.raw(), test.raw(), ref.size(), tolerance);
}

inline void printVerification(const std::string& label, const VerificationResult& res) {
    std::cout << "[" << label << "] Verification: "
              << (res.passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << " | Max Diff: " << std::scientific << std::setprecision(3) << res.max_abs_diff
              << " | RMSE: " << res.rmse
              << " | Rel L2: " << res.relative_l2_error
              << " | Mismatches: " << res.mismatched_elements << "\n";
}

} // namespace hpc
