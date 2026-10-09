#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>

namespace hpc {

// 2D Matrix container (Row-Major format)
class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, float initVal = 0.0f) : rows(r), cols(c), data(r * c, initVal) {}

    inline float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    inline const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    inline float* raw() { return data.data(); }
    inline const float* raw() const { return data.data(); }
    inline size_t size() const { return data.size(); }
    inline size_t byteSize() const { return data.size() * sizeof(float); }

    void randomize(float minVal = 0.0f, float maxVal = 1.0f, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(minVal, maxVal);
        for (auto& val : data) {
            val = dis(gen);
        }
    }
};

// 2D Image container (Grayscale float buffer in range [0.0, 255.0])
class Image {
public:
    int width;
    int height;
    int channels; // 1 for grayscale
    std::vector<float> data;

    Image() : width(0), height(0), channels(1) {}
    Image(int w, int h, int c = 1, float initVal = 0.0f)
        : width(w), height(h), channels(c), data(w * h * c, initVal) {}

    inline float& at(int y, int x, int c = 0) {
        return data[(y * width + x) * channels + c];
    }

    inline const float& at(int y, int x, int c = 0) const {
        return data[(y * width + x) * channels + c];
    }

    inline float* raw() { return data.data(); }
    inline const float* raw() const { return data.data(); }
    inline size_t size() const { return data.size(); }
    inline size_t byteSize() const { return data.size() * sizeof(float); }

    // Generate high-resolution synthetic image (e.g. 4K: 3840 x 2160)
    static Image createSynthetic4K(int w = 3840, int h = 2160) {
        Image img(w, h, 1);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Gradient pattern with sharp geometric frequency edges for filter testing
                float val = 128.0f + 64.0f * std::sin(x * 0.05f) + 63.0f * std::cos(y * 0.05f);
                // Inject sharp step-edge box in the center
                if (x >= w / 4 && x <= 3 * w / 4 && y >= h / 4 && y <= 3 * h / 4) {
                    val = ((x / 32) % 2 == (y / 32) % 2) ? 255.0f : 0.0f;
                }
                img.at(y, x) = std::max(0.0f, std::min(255.0f, val));
            }
        }
        return img;
    }
};

// 2D Convolution Kernel
struct Kernel {
    int size;       // Must be odd (e.g. 3, 5)
    int radius;     // size / 2
    std::vector<float> weights;

    Kernel(int kSize, const std::vector<float>& w)
        : size(kSize), radius(kSize / 2), weights(w) {
        if (weights.size() != static_cast<size_t>(kSize * kSize)) {
            throw std::invalid_argument("Kernel weights count does not match size*size");
        }
    }

    inline float get(int ky, int kx) const {
        return weights[ky * size + kx];
    }
};

} // namespace hpc
