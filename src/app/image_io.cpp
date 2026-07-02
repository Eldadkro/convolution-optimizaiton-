#include "kopt/image_io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <stdexcept>

namespace kopt {

std::vector<std::uint8_t> read_grayscale_image_u8(const std::string& path, int& width, int& height) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("failed to load image: " + path);
    }
    if (image.depth() != CV_8U || image.channels() != 1) {
        throw std::runtime_error("expected an 8-bit grayscale image: " + path);
    }

    width = image.cols;
    height = image.rows;

    if (image.isContinuous()) {
        return {image.data, image.data + image.total()};
    }

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        const auto* source_row = image.ptr<std::uint8_t>(y);
        auto* target_row = pixels.data() + static_cast<std::size_t>(y) * width;
        std::copy(source_row, source_row + width, target_row);
    }
    return pixels;
}

}  // namespace kopt
