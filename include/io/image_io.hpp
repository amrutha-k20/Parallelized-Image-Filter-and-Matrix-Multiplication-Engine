#pragma once

#include "common/types.hpp"
#include <string>

namespace hpc {

// Image I/O API
bool loadImage(const std::string& filepath, Image& image);
bool saveImage(const std::string& filepath, const Image& image);

// Direct Netpbm (PGM binary P5 format) I/O
bool savePGM(const std::string& filepath, const Image& image);
bool loadPGM(const std::string& filepath, Image& image);

} // namespace hpc
