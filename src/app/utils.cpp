#include "kopt/utils.hpp"

#include <filesystem>
#include <opencv2/core/hal/interface.h>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace kopt {

void save_image(const std::vector<std::uint8_t>& pixels, int width, int height, const std::string& filename) {
    std::filesystem::create_directories("results");
    const cv::Mat image(height, width, CV_8UC1, const_cast<std::uint8_t*>(pixels.data()), width);
    if (!cv::imwrite("results/" + filename, image, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
        throw std::runtime_error("failed to write results/" + filename);
    }
}

void save_image(const uint8_t* pixels, int width, int height, const std::string& filename) {
    std::filesystem::create_directories("results");
    const cv::Mat image(height, width, CV_8UC1, const_cast<std::uint8_t*>(pixels), width);
    if (!cv::imwrite("results/" + filename, image, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
        throw std::runtime_error("failed to write results/" + filename);
    }
}

void save_image(const float* pixels, int width, int height, const std::string& filename) {
    std::filesystem::create_directories("results");
    const cv::Mat image(height, width, CV_32FC1, const_cast<float*>(pixels), width* sizeof(float));
    if (!cv::imwrite("results/" + filename, image, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
        throw std::runtime_error("failed to write results/" + filename);
    }
}


} // namespace kopt
