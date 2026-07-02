#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kopt {

void save_image(const std::vector<std::uint8_t>& pixels, int width, int height, const std::string& filename);
void save_image(const uint8_t* pixels, int width, int height, const std::string& filename);
void save_image(const float* pixels, int width, int height, const std::string& filename);


} // namespace kopt
