#include "kopt/convolution.hpp"
#include "kopt/pgm_image.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kDefaultKernel[25] = {
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <image.pgm> <iterations> [cpu|cuda]\n";
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
        const std::string backend = argc == 4 ? argv[3] : "cpu";
        if (iterations <= 0) {
            throw std::runtime_error("iterations must be positive");
        }

        int width = 0;
        int height = 0;
        std::vector<kopt::Pixel> input_pixels = kopt::read_pgm_as_float(image_path, width, height);
        if (width != kopt::kImageWidth || height != kopt::kImageHeight) {
            std::cerr << "Warning: expected " << kopt::kImageWidth << "x" << kopt::kImageHeight
                      << ", got " << width << "x" << height << "\n";
        }

        std::vector<kopt::Pixel> output_pixels(input_pixels.size());
        const kopt::ConstImageView input{input_pixels.data(), width, height, width};
        const kopt::ImageView output{output_pixels.data(), width, height, width};

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (backend == "cpu") {
                kopt::convolve5x5_cpu(input, output, kDefaultKernel);
            } else if (backend == "cuda") {
#if KOPT_WITH_CUDA
                kopt::convolve5x5_cuda(input, output, kDefaultKernel);
#else
                throw std::runtime_error("CUDA backend was not built");
#endif
            } else {
                throw std::runtime_error("unknown backend: " + backend);
            }
        }
        const auto end = std::chrono::steady_clock::now();

        const auto elapsed = std::chrono::duration<double, std::milli>(end - start);
        std::cout << "backend=" << backend << "\n";
        std::cout << "iterations=" << iterations << "\n";
        std::cout << "total_ms=" << elapsed.count() << "\n";
        std::cout << "avg_ms=" << elapsed.count() / iterations << "\n";
        std::cout << "checksum_sample=" << output_pixels[output_pixels.size() / 2] << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
