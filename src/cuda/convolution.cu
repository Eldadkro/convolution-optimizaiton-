#include "kopt/convolution.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace kopt {
namespace {

__constant__ float kDeviceKernel[25];

__global__ void convolve5x5_kernel(const Pixel* input,
                                   Pixel* output,
                                   int width,
                                   int height,
                                   int input_stride,
                                   int output_stride) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -kKernelRadius; ky <= kKernelRadius; ++ky) {
        const int sy = min(max(y + ky, 0), height - 1);
        for (int kx = -kKernelRadius; kx <= kKernelRadius; ++kx) {
            const int sx = min(max(x + kx, 0), width - 1);
            const float coeff = kDeviceKernel[(ky + kKernelRadius) * kKernelSize + (kx + kKernelRadius)];
            sum += input[sy * input_stride + sx] * coeff;
        }
    }

    output[y * output_stride + x] = sum;
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

}  // namespace

void convolve5x5_cuda(ConstImageView input, ImageView output, const float kernel[25]) {
    const std::size_t input_bytes = static_cast<std::size_t>(input.stride) * input.height * sizeof(Pixel);
    const std::size_t output_bytes = static_cast<std::size_t>(output.stride) * output.height * sizeof(Pixel);

    Pixel* device_input = nullptr;
    Pixel* device_output = nullptr;

    check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc input");
    check_cuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc output");

    check_cuda(cudaMemcpy(device_input, input.data, input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy input");
    check_cuda(cudaMemcpyToSymbol(kDeviceKernel, kernel, 25 * sizeof(float)), "cudaMemcpyToSymbol kernel");

    const dim3 block(32, 8);
    const dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);
    convolve5x5_kernel<<<grid, block>>>(device_input,
                                        device_output,
                                        input.width,
                                        input.height,
                                        input.stride,
                                        output.stride);
    check_cuda(cudaGetLastError(), "convolve5x5_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "convolve5x5_kernel sync");

    check_cuda(cudaMemcpy(output.data, device_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output");

    cudaFree(device_output);
    cudaFree(device_input);
}

}  // namespace kopt
