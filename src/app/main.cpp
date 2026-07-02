#include "kopt/convolution.hpp"
#include "kopt/filter.hpp"
#include "kopt/image_io.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

}  // namespace

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
            std::cerr << "Warning: expected " << kopt::kImageWidth << "x" << kopt::kImageHeight
                      << ", got " << width << "x" << height << "\n";
        }

        std::vector<std::uint8_t> lpf_pixels(input_pixels.size());
        std::vector<std::uint8_t> hpf_pixels(input_pixels.size());
        const kopt::ImageViewU8 input{input_pixels.data(), width, height, width};
        const kopt::MutableImageViewU8 lpf_output{lpf_pixels.data(), width, height, width};
        const kopt::MutableImageViewU8 hpf_output{hpf_pixels.data(), width, height, width};
        kopt::FilterEngine engine({backend, 2});

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
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
