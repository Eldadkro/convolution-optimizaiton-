#include "kopt/filter.hpp"

#include "kopt/convolution.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace kopt {
namespace {

constexpr float kDefaultKernel[25] = {
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
};

void validate_view(ImageViewU8 view, const char* name) {
    if (view.data == nullptr) {
        throw std::invalid_argument(std::string(name) + ".data is null");
    }
    if (view.width <= 0 || view.height <= 0) {
        throw std::invalid_argument(std::string(name) + " dimensions must be positive");
    }
    if (view.stride_bytes < view.width) {
        throw std::invalid_argument(std::string(name) + ".stride_bytes is smaller than width");
    }
}

void validate_view(MutableImageViewU8 view, const char* name) {
    validate_view(ImageViewU8{view.data, view.width, view.height, view.stride_bytes}, name);
}

void validate_same_shape(ImageViewU8 input, MutableImageViewU8 output, const char* name) {
    if (input.width != output.width || input.height != output.height) {
        throw std::invalid_argument(std::string(name) + " dimensions do not match input");
    }
}

std::uint8_t to_u8(float value) {
    const float scaled = std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f);
    return static_cast<std::uint8_t>(scaled);
}

}  // namespace

class FilterEngine::Impl {
public:
    explicit Impl(FilterConfig config) : config_(config), active_backend_(resolve_backend(config.backend)) {
        if (config_.radius != kKernelRadius) {
            throw std::invalid_argument("only radius 2 is supported");
        }
    }

    void run(ImageViewU8 input, MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output) {
        validate_view(input, "input");
        validate_view(lpf_output, "lpf_output");
        validate_view(hpf_output, "hpf_output");
        validate_same_shape(input, lpf_output, "lpf_output");
        validate_same_shape(input, hpf_output, "hpf_output");

        const std::size_t pixel_count = static_cast<std::size_t>(input.width) * input.height;
        std::vector<Pixel> input_pixels(pixel_count);
        std::vector<Pixel> lpf_pixels(pixel_count);

        for (int y = 0; y < input.height; ++y) {
            const auto* row = input.data + static_cast<std::ptrdiff_t>(y) * input.stride_bytes;
            for (int x = 0; x < input.width; ++x) {
                input_pixels[static_cast<std::size_t>(y) * input.width + x] =
                    static_cast<Pixel>(row[x]) / 255.0f;
            }
        }

        const ConstImageView input_view{input_pixels.data(), input.width, input.height, input.width};
        const ImageView lpf_view{lpf_pixels.data(), input.width, input.height, input.width};

        if (active_backend_ == Backend::Cpu) {
            convolve5x5_cpu(input_view, lpf_view, kDefaultKernel);
        } else if (active_backend_ == Backend::Cuda) {
#if KOPT_WITH_CUDA
            convolve5x5_cuda(input_view, lpf_view, kDefaultKernel);
#else
            throw std::runtime_error("CUDA backend was not built");
#endif
        } else {
            throw std::runtime_error("unresolved filter backend");
        }

        for (int y = 0; y < input.height; ++y) {
            auto* lpf_row = lpf_output.data + static_cast<std::ptrdiff_t>(y) * lpf_output.stride_bytes;
            auto* hpf_row = hpf_output.data + static_cast<std::ptrdiff_t>(y) * hpf_output.stride_bytes;
            for (int x = 0; x < input.width; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * input.width + x;
                const float lpf = lpf_pixels[index];
                lpf_row[x] = to_u8(lpf);
                hpf_row[x] = to_u8(input_pixels[index] - lpf + 0.5f);
            }
        }
    }

    [[nodiscard]] Backend active_backend() const noexcept {
        return active_backend_;
    }

private:
    static Backend resolve_backend(Backend requested) {
        if (requested == Backend::Auto) {
#if KOPT_WITH_CUDA
            return Backend::Cuda;
#else
            return Backend::Cpu;
#endif
        }
        return requested;
    }

    FilterConfig config_;
    Backend active_backend_;
};

FilterEngine::FilterEngine(FilterConfig config) : impl_(std::make_unique<Impl>(config)) {}

FilterEngine::~FilterEngine() = default;

FilterEngine::FilterEngine(FilterEngine&&) noexcept = default;

FilterEngine& FilterEngine::operator=(FilterEngine&&) noexcept = default;

void FilterEngine::run(ImageViewU8 input, MutableImageViewU8 lpf_output, MutableImageViewU8 hpf_output) {
    impl_->run(input, lpf_output, hpf_output);
}

Backend FilterEngine::active_backend() const noexcept {
    return impl_->active_backend();
}

}  // namespace kopt
