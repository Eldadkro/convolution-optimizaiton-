#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kopt {

std::vector<std::uint8_t> read_grayscale_image_u8(const std::string& path, int& width, int& height);

}  // namespace kopt
