#include "io/image_io.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

#ifdef HPC_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace hpc {

// Binary PGM (P5 format) Writer
bool savePGM(const std::string& filepath, const Image& image) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << "\n";
        return false;
    }

    file << "P5\n" << image.width << " " << image.height << "\n255\n";

    std::vector<uint8_t> buffer(image.width * image.height);
    for (size_t i = 0; i < buffer.size(); ++i) {
        float val = image.data[i];
        buffer[i] = static_cast<uint8_t>(std::min(std::max(val, 0.0f), 255.0f));
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    return true;
}

// Binary PGM (P5 format) Reader
bool loadPGM(const std::string& filepath, Image& image) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filepath << "\n";
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "Invalid PGM format (expected P5): " << filepath << "\n";
        return false;
    }

    int width = 0, height = 0, maxVal = 0;
    file >> width >> height >> maxVal;
    file.ignore(1); // skip newline

    image = Image(width, height, 1);
    std::vector<uint8_t> buffer(width * height);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (size_t i = 0; i < buffer.size(); ++i) {
        image.data[i] = static_cast<float>(buffer[i]);
    }

    return true;
}

bool loadImage(const std::string& filepath, Image& image) {
#ifdef HPC_USE_OPENCV
    cv::Mat mat = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (mat.empty()) {
        std::cerr << "OpenCV failed to read image: " << filepath << ", falling back to PGM reader\n";
        return loadPGM(filepath, image);
    }
    image = Image(mat.cols, mat.rows, 1);
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            image.at(y, x) = static_cast<float>(mat.at<uint8_t>(y, x));
        }
    }
    return true;
#else
    return loadPGM(filepath, image);
#endif
}

bool saveImage(const std::string& filepath, const Image& image) {
#ifdef HPC_USE_OPENCV
    cv::Mat mat(image.height, image.width, CV_8UC1);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float val = image.at(y, x);
            mat.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(std::max(val, 0.0f), 255.0f));
        }
    }
    return cv::imwrite(filepath, mat);
#else
    return savePGM(filepath, image);
#endif
}

} // namespace hpc
