#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

namespace kopt {

inline constexpr int kImageWidth = 2048;
inline constexpr int kImageHeight = 1024;
inline constexpr int kKernelSize = 5;
inline constexpr int kKernelRadius = kKernelSize / 2;

using Pixel = float;

struct ImageView {
    Pixel* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

struct ConstImageView {
    const Pixel* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

void convolve5x5_cpu(ConstImageView input, ImageView output, const float kernel[25]);

void convolve5x5_cpu_opencv(ConstImageView input, ImageView output, const float kernel[25]);

void convolve5x5_cuda(ConstImageView input, ImageView output, const float kernel[25]);

}  // namespace kopt
