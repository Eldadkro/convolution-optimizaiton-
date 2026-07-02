#pragma once

#include <string>
#include <vector>

#include "kopt/convolution.hpp"

namespace kopt {

std::vector<Pixel> read_pgm_as_float(const std::string& path, int& width, int& height);

}  // namespace kopt
