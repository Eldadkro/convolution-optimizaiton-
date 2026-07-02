#include "kopt/convolution.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>

#include "kopt/utils.hpp"

namespace kopt {

std::vector<std::uint32_t> integral_u8_to_u32(ConstImageViewU8 input) {
    if (input.data == nullptr) {
        throw std::invalid_argument("input.data is null");
    }
    if (input.width <= 0 || input.height <= 0) {
        throw std::invalid_argument("input dimensions must be positive");
    }
    if (input.stride_bytes < input.width) {
        throw std::invalid_argument("input.stride_bytes is smaller than width");
    }

    const int integral_width = input.width + 1;
    std::vector<std::uint32_t> integral(static_cast<std::size_t>(integral_width) * (input.height + 1), 0);

    for (int y = 0; y < input.height; ++y) {
        const auto* input_row = input.data + static_cast<std::ptrdiff_t>(y) * input.stride_bytes;
        auto* integral_row = integral.data() + static_cast<std::size_t>(y + 1) * integral_width;
        const auto* integral_prev_row = integral.data() + static_cast<std::size_t>(y) * integral_width;

        std::uint32_t row_sum = 0;
        for (int x = 0; x < input.width; ++x) {
            row_sum += input_row[x];
            integral_row[x + 1] = integral_prev_row[x + 1] + row_sum;
        }
    }

    return integral;
}

void convolve5x5_cpu(ConstImageView input, ImageView output, const float kernel[25]) {
    // save_image(input.data, input.width, input.height, "inputconv.png");
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            float sum = 0.0f;

            for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
                const int sy = std::clamp(y + ky, 0, input.height - 1);
                for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
                    const int sx = std::clamp(x + kx, 0, input.width - 1);
                    const float coeff = kernel[(ky + kKernelRadius) * kKernelSize + (kx + kKernelRadius)];
                    sum += input.data[sy * input.stride + sx] * coeff;
                }
            }

            output.data[y * output.stride + x] = sum;
        }
    }
}

void convolve5x5_cpu_opencv(ConstImageView input, ImageView output, const float kernel[25]) {
    const cv::Mat input_mat(input.height, input.width, CV_32FC1, const_cast<Pixel*>(input.data),
                            static_cast<std::size_t>(input.stride) * sizeof(Pixel));
    cv::Mat output_mat(output.height, output.width, CV_32FC1, output.data,
                       static_cast<std::size_t>(output.stride) * sizeof(Pixel));
    const cv::Mat kernel_mat(kKernelSize, kKernelSize, CV_32FC1, const_cast<float*>(kernel));

    cv::filter2D(input_mat, output_mat, CV_32F, kernel_mat, cv::Point(-1, -1), 0.0, cv::BORDER_DEFAULT);
}

void convolve5x5_box_cpu(ConstImageView input, ImageView output) {
    const int padded_width = input.width + 2 * kKernelRadius;
    const int padded_height = input.height + 2 * kKernelRadius;
    const int integral_width = padded_width + 1;
    std::vector<double> integral(static_cast<std::size_t>(integral_width) * (padded_height + 1), 0.0);

    for (int y = 0; y < padded_height; ++y) {
        const int sy = std::clamp(y - kKernelRadius, 0, input.height - 1);
        double row_sum = 0.0;
        for (int x = 0; x < padded_width; ++x) {
            const int sx = std::clamp(x - kKernelRadius, 0, input.width - 1);
            row_sum += input.data[sy * input.stride + sx];
            integral[static_cast<std::size_t>(y + 1) * integral_width + x + 1] =
                integral[static_cast<std::size_t>(y) * integral_width + x + 1] + row_sum;
        }
    }

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            const int x0 = x;
            const int y0 = y;
            const int x1 = x + kKernelSize;
            const int y1 = y + kKernelSize;
            const double sum = integral[static_cast<std::size_t>(y1) * integral_width + x1] -
                               integral[static_cast<std::size_t>(y0) * integral_width + x1] -
                               integral[static_cast<std::size_t>(y1) * integral_width + x0] +
                               integral[static_cast<std::size_t>(y0) * integral_width + x0];
            output.data[y * output.stride + x] = static_cast<Pixel>(sum / (kKernelSize * kKernelSize));
        }
    }
}

void convolve5x5_box_cpu_opencv_integral(ConstImageView input, ImageView output) {
    const cv::Mat input_mat(input.height, input.width, CV_32FC1, const_cast<Pixel*>(input.data),
                            static_cast<std::size_t>(input.stride) * sizeof(Pixel));

    cv::Mat padded;
    cv::copyMakeBorder(input_mat, padded, kKernelRadius, kKernelRadius, kKernelRadius, kKernelRadius,
                       cv::BORDER_REPLICATE);

    cv::Mat integral;
    cv::integral(padded, integral, CV_64F);

    constexpr double scale = 1.0 / static_cast<double>(kKernelSize * kKernelSize);
    for (int y = 0; y < input.height; ++y) {
        auto* output_row = output.data + static_cast<std::ptrdiff_t>(y) * output.stride;
        const double* top = integral.ptr<double>(y);
        const double* bottom = integral.ptr<double>(y + kKernelSize);

        for (int x = 0; x < input.width; ++x) {
            const double sum = bottom[x + kKernelSize] - top[x + kKernelSize] - bottom[x] + top[x];
            output_row[x] = static_cast<Pixel>(sum * scale);
        }
    }
}


void convolve5x5_box_cpu_opencv_separable(ConstImageView input, ImageView output) {
    constexpr float coeff = 1.0f / static_cast<float>(5);
    static cv::Mat kernelSepBox = (cv::Mat_<float>(1, kKernelSize) << coeff, coeff, coeff, coeff, coeff);
    const cv::Mat input_mat(input.height, input.width, CV_32FC1, const_cast<Pixel*>(input.data),
                            static_cast<std::size_t>(input.stride) * sizeof(Pixel));
    cv::Mat output_mat(output.height, output.width, CV_32FC1, output.data,
                       static_cast<std::size_t>(output.stride) * sizeof(Pixel));

    cv::sepFilter2D(input_mat, output_mat, CV_32F, kernelSepBox, kernelSepBox, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
}

} // namespace kopt
