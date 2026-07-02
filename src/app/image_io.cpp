#include "kopt/image_io.hpp"

#include <cstdint>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace kopt {

std::vector<std::uint8_t> read_grayscale_image_u8(const std::string& path, int& width, int& height) {
    const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("failed to load image: " + path);
    }
    if (image.depth() != CV_8U) {
        throw std::runtime_error("expected an 8-bit image: " + path);
    }

    cv::Mat grayscale;
    if (image.channels() == 1) {
        grayscale = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, grayscale, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("expected a grayscale, RGB, or RGBA image: " + path);
    }

    width = grayscale.cols;
    height = grayscale.rows;

    if (grayscale.isContinuous()) {
        return {grayscale.data, grayscale.data + grayscale.total()};
    }

    const cv::Mat continuous = grayscale.clone();
    return {continuous.data, continuous.data + continuous.total()};
}

}  // namespace kopt
