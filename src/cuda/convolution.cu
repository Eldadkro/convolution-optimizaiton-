#include "kopt/convolution.hpp"

#include <cuda_runtime.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace kopt {
namespace {

__constant__ float kDeviceKernel[25];

__global__ void convolve5x5_kernel(const Pixel* input, Pixel* output, int width, int height, int input_stride,
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

__global__ void convolve5x5_u8_kernel(const std::uint8_t* input, Pixel* lpf, int width, int height,
                                      std::ptrdiff_t input_stride_bytes, int lpf_stride) {
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
            sum += static_cast<float>(input[sy * input_stride_bytes + sx]) * coeff;
        }
    }

    lpf[y * lpf_stride + x] = sum;
}

__device__ std::uint8_t to_u8_device(float value) {
    const float clamped = min(max(value, 0.0f), 255.0f);
    return static_cast<std::uint8_t>(roundf(clamped));
}

__global__ void write_filter_outputs_kernel(const std::uint8_t* input, const std::uint8_t* lpf_output,
                                            std::uint8_t* hpf_output, int width, int height,
                                            std::ptrdiff_t input_stride_bytes, std::ptrdiff_t lpf_output_stride_bytes,
                                            std::ptrdiff_t hpf_output_stride_bytes) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float input_value = static_cast<float>(input[y * input_stride_bytes + x]);
    const float lpf_value = static_cast<float>(lpf_output[y * lpf_output_stride_bytes + x]);
    hpf_output[y * hpf_output_stride_bytes + x] = to_u8_device(input_value - lpf_value + 0.5f);
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void check_npp(NppStatus result, const char* operation) {
    if (result < 0) {
        throw std::runtime_error(std::string(operation) + ": NPP error " + std::to_string(result));
    }
}

} // namespace

CudaConvolutionState::~CudaConvolutionState() {
    release();
}

CudaConvolutionState::CudaConvolutionState(CudaConvolutionState&& other) noexcept {
    *this = std::move(other);
}

CudaConvolutionState& CudaConvolutionState::operator=(CudaConvolutionState&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();
    device_input = std::exchange(other.device_input, nullptr);
    device_lpf = std::exchange(other.device_lpf, nullptr);
    device_lpf_output = std::exchange(other.device_lpf_output, nullptr);
    device_hpf_output = std::exchange(other.device_hpf_output, nullptr);
    stream = std::exchange(other.stream, nullptr);
    width = std::exchange(other.width, 0);
    height = std::exchange(other.height, 0);
    input_stride_bytes = std::exchange(other.input_stride_bytes, 0);
    device_lpf_stride = std::exchange(other.device_lpf_stride, 0);
    lpf_output_stride_bytes = std::exchange(other.lpf_output_stride_bytes, 0);
    hpf_output_stride_bytes = std::exchange(other.hpf_output_stride_bytes, 0);
    input_bytes = std::exchange(other.input_bytes, 0);
    device_lpf_bytes = std::exchange(other.device_lpf_bytes, 0);
    lpf_output_bytes = std::exchange(other.lpf_output_bytes, 0);
    hpf_output_bytes = std::exchange(other.hpf_output_bytes, 0);
    return *this;
}

void CudaConvolutionState::setup(ImageViewU8 input, MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output,
                                 const float kernel[25]) {
    release();
    width = input.width;
    height = input.height;
    input_stride_bytes = input.stride_bytes;
    device_lpf_stride = input.width;
    lpf_output_stride_bytes = lpf_output.stride_bytes;
    hpf_output_stride_bytes = hpf_output.stride_bytes;
    input_bytes = static_cast<std::size_t>(input.stride_bytes) * input.height;
    device_lpf_bytes = static_cast<std::size_t>(device_lpf_stride) * input.height * sizeof(Pixel);
    lpf_output_bytes = static_cast<std::size_t>(lpf_output.stride_bytes) * lpf_output.height;
    hpf_output_bytes = static_cast<std::size_t>(hpf_output.stride_bytes) * hpf_output.height;

    try {
        check_cuda(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream)), "cudaStreamCreate");
        check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&device_lpf, device_lpf_bytes), "cudaMalloc lpf");
        check_cuda(cudaMalloc(&device_lpf_output, lpf_output_bytes), "cudaMalloc lpf output");
        check_cuda(cudaMalloc(&device_hpf_output, hpf_output_bytes), "cudaMalloc hpf output");
    } catch (...) {
        release();
        throw;
    }

    check_cuda(cudaMemcpyToSymbolAsync(kDeviceKernel, kernel, 25 * sizeof(float), 0, cudaMemcpyHostToDevice,
                                       reinterpret_cast<cudaStream_t>(stream)),
               "cudaMemcpyToSymbol kernel");
}

void CudaConvolutionState::release() noexcept {
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cuda_stream != nullptr) {
        cudaStreamSynchronize(cuda_stream);
    }
    if (device_hpf_output != nullptr) {
        cudaFree(device_hpf_output);
    }
    if (device_lpf_output != nullptr) {
        cudaFree(device_lpf_output);
    }
    if (device_lpf != nullptr) {
        cudaFree(device_lpf);
    }
    if (device_input != nullptr) {
        cudaFree(device_input);
    }
    if (cuda_stream != nullptr) {
        cudaStreamDestroy(cuda_stream);
    }
    device_input = nullptr;
    device_lpf = nullptr;
    device_lpf_output = nullptr;
    device_hpf_output = nullptr;
    stream = nullptr;
    width = 0;
    height = 0;
    input_stride_bytes = 0;
    device_lpf_stride = 0;
    lpf_output_stride_bytes = 0;
    hpf_output_stride_bytes = 0;
    input_bytes = 0;
    device_lpf_bytes = 0;
    lpf_output_bytes = 0;
    hpf_output_bytes = 0;
}

void CudaConvolutionState::copy_input(ImageViewU8 input) {
    if (device_input == nullptr) {
        throw std::logic_error("CUDA convolution state is not initialized");
    }
    if (input.width != width || input.height != height || input.stride_bytes != input_stride_bytes) {
        throw std::invalid_argument("CUDA input view does not match allocated state");
    }

    check_cuda(cudaMemcpyAsync(device_input, input.data, input_bytes, cudaMemcpyHostToDevice,
                               reinterpret_cast<cudaStream_t>(stream)),
               "cudaMemcpy input");
}

void CudaConvolutionState::run() {
    if (device_input == nullptr || device_lpf_output == nullptr || device_hpf_output == nullptr) {
        throw std::logic_error("CUDA convolution state is not initialized");
    }
    // const dim3 block(32, 8);
    // const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // convolve5x5_u8_kernel<<<grid, block>>>(device_input, device_lpf, width, height, input_stride_bytes,
    //                                        device_lpf_stride);
    run_npp_box_filter();

    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    write_filter_outputs_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        device_input, device_lpf_output, device_hpf_output, width, height, input_stride_bytes, lpf_output_stride_bytes,
        hpf_output_stride_bytes);
    check_cuda(cudaGetLastError(), "write_filter_outputs_kernel launch");
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "CUDA filter kernels sync");
}

void CudaConvolutionState::run_npp_box_filter() {
    if (device_input == nullptr || device_lpf_output == nullptr) {
        throw std::logic_error("CUDA convolution state is not initialized");
    }

    const NppiSize source_size{width, height};
    const NppiPoint source_offset{0, 0};
    const NppiSize roi_size{width, height};
    const NppiSize mask_size{kKernelSize, kKernelSize};
    const NppiPoint anchor{kKernelRadius, kKernelRadius};

    NppStreamContext stream_context;
    check_npp(nppGetStreamContext(&stream_context), "nppGetStreamContext");
    stream_context.hStream = reinterpret_cast<cudaStream_t>(stream);

    check_npp(nppiFilterBoxBorder_8u_C1R_Ctx(device_input, static_cast<Npp32s>(input_stride_bytes), source_size,
                                             source_offset, device_lpf_output,
                                             static_cast<Npp32s>(lpf_output_stride_bytes), roi_size, mask_size, anchor,
                                             NPP_BORDER_REPLICATE, stream_context),
              "nppiFilterBoxBorder_8u_C1R");
}

void CudaConvolutionState::copy_output(MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output) {
    if (device_lpf_output == nullptr || device_hpf_output == nullptr) {
        throw std::logic_error("CUDA convolution state is not initialized");
    }
    if (lpf_output.width != width || lpf_output.height != height ||
        lpf_output.stride_bytes != lpf_output_stride_bytes || hpf_output.width != width ||
        hpf_output.height != height || hpf_output.stride_bytes != hpf_output_stride_bytes) {
        throw std::invalid_argument("CUDA output views do not match allocated state");
    }

    check_cuda(cudaMemcpyAsync(lpf_output.data, device_lpf_output, lpf_output_bytes, cudaMemcpyDeviceToHost,
                               reinterpret_cast<cudaStream_t>(stream)),
               "cudaMemcpy lpf output");
    check_cuda(cudaMemcpyAsync(hpf_output.data, device_hpf_output, hpf_output_bytes, cudaMemcpyDeviceToHost,
                               reinterpret_cast<cudaStream_t>(stream)),
               "cudaMemcpy hpf output");
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaMemcpy output sync");
}

} // namespace kopt
