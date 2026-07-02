#include "kopt/convolution.hpp"
#include "kopt/filter.hpp"
#include "kopt/image_io.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <image> <iterations> [auto|cpu|cuda]\n";
}

kopt::Backend parse_backend(const std::string& value) {
    if (value == "auto") {
        return kopt::Backend::Auto;
    }
    if (value == "cpu") {
        return kopt::Backend::Cpu;
    }
    if (value == "cuda") {
        return kopt::Backend::Cuda;
    }
    throw std::runtime_error("unknown backend: " + value);
}

const char* backend_name(kopt::Backend backend) {
    switch (backend) {
    case kopt::Backend::Auto:
        return "auto";
    case kopt::Backend::Cpu:
        return "cpu";
    case kopt::Backend::Cuda:
        return "cuda";
    }
    return "unknown";
}

void save_lpf_image(const std::vector<std::uint8_t>& pixels, int width, int height) {
    std::filesystem::create_directories("results");
    const cv::Mat image(height, width, CV_8UC1, const_cast<std::uint8_t*>(pixels.data()), width);
    if (!cv::imwrite("results/lpf.png", image, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
        throw std::runtime_error("failed to write results/lpf.png");
    }
}

void save_hpf_image(const std::vector<std::uint8_t>& pixels, int width, int height) {
    std::filesystem::create_directories("results");
    const cv::Mat image(height, width, CV_8UC1, const_cast<std::uint8_t*>(pixels.data()), width);
    if (!cv::imwrite("results/hpf.png", image, {cv::IMWRITE_PNG_COMPRESSION, 0})) {
        throw std::runtime_error("failed to write results/hpf.png");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string image_path = argv[1];
        const int iterations = std::stoi(argv[2]);
        const std::string backend_arg = argc == 4 ? argv[3] : "auto";
        const kopt::Backend backend = parse_backend(backend_arg);
        if (iterations <= 0) {
            throw std::runtime_error("iterations must be positive");
        }

        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> input_pixels = kopt::read_grayscale_image_u8(image_path, width, height);
        if (width != kopt::kImageWidth || height != kopt::kImageHeight) {
            std::cerr << "Warning: expected " << kopt::kImageWidth << "x" << kopt::kImageHeight << ", got " << width
                      << "x" << height << "\n";
        }

        std::vector<std::uint8_t> lpf_pixels(input_pixels.size());
        std::vector<std::uint8_t> hpf_pixels(input_pixels.size());
        const kopt::ImageViewU8 input{input_pixels.data(), width, height, width};
        const kopt::MutableImageViewU8 lpf_output{lpf_pixels.data(), width, height, width};
        const kopt::MutableImageViewU8 hpf_output{hpf_pixels.data(), width, height, width};
        kopt::FilterEngine engine;
        engine.init({backend, 2});
        engine.setup(width, height);

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            engine.run(input, lpf_output, hpf_output);
        }
        const auto end = std::chrono::steady_clock::now();

        const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        std::cout << "backend=" << backend_name(engine.active_backend()) << "\n";
        std::cout << "iterations=" << iterations << "\n";
        std::cout << "total_ms=" << elapsed.count() << "\n";
        std::cout << "avg_ms=" << elapsed.count() / iterations << "\n";
        std::cout << "lpf_sample=" << static_cast<int>(lpf_pixels[lpf_pixels.size() / 2]) << "\n";
        std::cout << "hpf_sample=" << static_cast<int>(hpf_pixels[hpf_pixels.size() / 2]) << "\n";

        save_lpf_image(lpf_pixels, width, height);
        save_hpf_image(hpf_pixels, width, height);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
