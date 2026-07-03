#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>

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

struct ImageViewU8 {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride_bytes = 0;
};

struct MutableImageViewU8 {
    std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride_bytes = 0;
};

using ConstImageViewU8 = ImageViewU8;

std::vector<std::uint32_t> integral_u8_to_u32(ConstImageViewU8 input);

void convolve5x5_cpu(ConstImageView input, ImageView output, const float kernel[25]);

void convolve5x5_box_cpu(ConstImageView input, ImageView output);

void convolve5x5_box_cpu_opencv_integral(ConstImageView input, ImageView output);

void convolve5x5_box_cpu_opencv_separable(ConstImageView input, ImageView output);

void convolve5x5_cpu_opencv(ConstImageView input, ImageView output, const float kernel[25]);

struct CudaConvolutionState {
    std::uint8_t* device_input = nullptr;
    Pixel* device_lpf = nullptr;
    std::uint8_t* device_lpf_output = nullptr;
    std::uint8_t* device_hpf_output = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t input_stride_bytes = 0;
    int device_lpf_stride = 0;
    std::ptrdiff_t lpf_output_stride_bytes = 0;
    std::ptrdiff_t hpf_output_stride_bytes = 0;
    std::size_t input_bytes = 0;
    std::size_t device_lpf_bytes = 0;
    std::size_t lpf_output_bytes = 0;
    std::size_t hpf_output_bytes = 0;

    CudaConvolutionState() = default;
    ~CudaConvolutionState();

    CudaConvolutionState(const CudaConvolutionState&) = delete;
    CudaConvolutionState& operator=(const CudaConvolutionState&) = delete;

    CudaConvolutionState(CudaConvolutionState&& other) noexcept;
    CudaConvolutionState& operator=(CudaConvolutionState&& other) noexcept;

    void setup(ImageViewU8 input, MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output, const float kernel[25]);

    void release() noexcept;

    void copy_input(ImageViewU8 input);

    void run();

    void copy_output(MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output);
};


}  // namespace kopt
