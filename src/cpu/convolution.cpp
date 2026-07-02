#include "kopt/convolution.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>

#include "kopt/utils.hpp"

namespace kopt {

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

} // namespace kopt
