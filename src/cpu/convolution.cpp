#include "kopt/convolution.hpp"

#include <algorithm>

namespace kopt {

void convolve5x5_cpu(ConstImageView input, ImageView output, const float kernel[25]) {
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

}  // namespace kopt
