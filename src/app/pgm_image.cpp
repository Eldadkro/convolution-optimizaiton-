#include "kopt/pgm_image.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace kopt {
namespace {

std::string read_token(std::istream& input) {
    std::string token;
    while (input >> token) {
        if (!token.empty() && token[0] == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        return token;
    }
    throw std::runtime_error("Unexpected end of PGM header");
}

}  // namespace

std::vector<Pixel> read_pgm_as_float(const std::string& path, int& width, int& height) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open image: " + path);
    }

    const std::string magic = read_token(input);
    if (magic != "P5") {
        throw std::runtime_error("Only binary PGM P5 images are supported");
    }

    width = std::stoi(read_token(input));
    height = std::stoi(read_token(input));
    const int max_value = std::stoi(read_token(input));
    if (max_value <= 0 || max_value > 255) {
        throw std::runtime_error("Only 8-bit PGM images are supported");
    }

    input.get();

    std::vector<unsigned char> bytes(static_cast<std::size_t>(width) * height);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("PGM image data is shorter than expected");
    }

    std::vector<Pixel> pixels(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        pixels[i] = static_cast<Pixel>(bytes[i]) / static_cast<Pixel>(max_value);
    }
    return pixels;
}

}  // namespace kopt
