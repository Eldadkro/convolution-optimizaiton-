#include "kopt/image_io.hpp"

#include <cstdint>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

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

    throw std::runtime_error("image isn't Continuous");
    return std::vector<uint8_t>();
}

}  // namespace kopt
