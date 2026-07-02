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
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  16.0f / 273.0f, 26.0f / 273.0f,
    16.0f / 273.0f, 4.0f / 273.0f,  7.0f / 273.0f,  26.0f / 273.0f,
    41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,  4.0f / 273.0f,
    16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f,
    1.0f / 273.0f,
};

void validate_view(ImageViewU8 view, const char *name) {
  if (view.data == nullptr) {
    throw std::invalid_argument(std::string(name) + ".data is null");
  }
  if (view.width <= 0 || view.height <= 0) {
    throw std::invalid_argument(std::string(name) +
                                " dimensions must be positive");
  }
  if (view.stride_bytes < view.width) {
    throw std::invalid_argument(std::string(name) +
                                ".stride_bytes is smaller than width");
  }
}

void validate_view(MutableImageViewU8 view, const char *name) {
  validate_view(
      ImageViewU8{view.data, view.width, view.height, view.stride_bytes}, name);
}

void validate_same_shape(ImageViewU8 input, MutableImageViewU8 output,
                         const char *name) {
  if (input.width != output.width || input.height != output.height) {
    throw std::invalid_argument(std::string(name) +
                                " dimensions do not match input");
  }
}

std::uint8_t to_u8(float value) {
  const float scaled = std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f);
  return static_cast<std::uint8_t>(scaled);
}

} // namespace

class FilterEngine::Impl {
public:
  Impl() = default;

  void init(FilterConfig config) {
    config_ = config;
    if (config_.radius != kKernelRadius) {
      throw std::invalid_argument("only radius 2 is supported");
    }
    active_backend_ = resolve_backend(config.backend);
    initialized_ = true;
    setup_ = false;
    width_ = 0;
    height_ = 0;
    input_pixels_.clear();
    lpf_pixels_.clear();
  }

  void setup(int width, int height) {
    if (!initialized_) {
      throw std::logic_error("FilterEngine must be initialized before setup");
    }
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument("setup dimensions must be positive");
    }

    width_ = width;
    height_ = height;
    const std::size_t pixel_count = static_cast<std::size_t>(width_) * height_;
    input_pixels_.assign(pixel_count, 0.0f);
    lpf_pixels_.assign(pixel_count, 0.0f);
    setup_ = true;
  }

  void run(ImageViewU8 input, MutableImageViewU8 lpf_output,
           MutableImageViewU8 hpf_output) {
    if (!initialized_) {
      throw std::logic_error("FilterEngine is not initialized");
    }
    validate_view(input, "input");
    validate_view(lpf_output, "lpf_output");
    validate_view(hpf_output, "hpf_output");
    validate_same_shape(input, lpf_output, "lpf_output");
    validate_same_shape(input, hpf_output, "hpf_output");

    if (!setup_) {
      setup(input.width, input.height);
    }
    if (input.width != width_ || input.height != height_) {
      throw std::invalid_argument(
          "input dimensions do not match FilterEngine setup");
    }

    for (int y = 0; y < input.height; ++y) {
      const auto *row =
          input.data + static_cast<std::ptrdiff_t>(y) * input.stride_bytes;
      for (int x = 0; x < input.width; ++x) {
        input_pixels_[static_cast<std::size_t>(y) * input.width + x] =
            static_cast<Pixel>(row[x]) / 255.0f;
      }
    }

    const ConstImageView input_view{input_pixels_.data(), input.width,
                                    input.height, input.width};
    const ImageView lpf_view{lpf_pixels_.data(), input.width, input.height,
                             input.width};

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
      auto *lpf_row = lpf_output.data +
                      static_cast<std::ptrdiff_t>(y) * lpf_output.stride_bytes;
      auto *hpf_row = hpf_output.data +
                      static_cast<std::ptrdiff_t>(y) * hpf_output.stride_bytes;
      for (int x = 0; x < input.width; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * input.width + x;
        const float lpf = lpf_pixels_[index];
        lpf_row[x] = to_u8(lpf);
        hpf_row[x] = to_u8(input_pixels_[index] - lpf + 0.5f);
      }
    }
  }

  [[nodiscard]] Backend active_backend() const noexcept {
    return active_backend_;
  }

  [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

  [[nodiscard]] bool is_setup() const noexcept { return setup_; }

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

  FilterConfig config_{};
  Backend active_backend_ = Backend::Auto;
  bool initialized_ = false;
  bool setup_ = false;
  int width_ = 0;
  int height_ = 0;
  std::vector<Pixel> input_pixels_;
  std::vector<Pixel> lpf_pixels_;
};

FilterEngine::FilterEngine() : impl_(std::make_unique<Impl>()) {}

FilterEngine::FilterEngine(FilterConfig config) : FilterEngine() {
  init(config);
}

FilterEngine::~FilterEngine() = default;

FilterEngine::FilterEngine(FilterEngine &&) noexcept = default;

FilterEngine &FilterEngine::operator=(FilterEngine &&) noexcept = default;

void FilterEngine::init(FilterConfig config) { impl_->init(config); }

void FilterEngine::setup(int width, int height) { impl_->setup(width, height); }

void FilterEngine::run(ImageViewU8 input, MutableImageViewU8 lpf_output,
                       MutableImageViewU8 hpf_output) {
  impl_->run(input, lpf_output, hpf_output);
}

Backend FilterEngine::active_backend() const noexcept {
  return impl_->active_backend();
}

bool FilterEngine::is_initialized() const noexcept {
  return impl_->is_initialized();
}

bool FilterEngine::is_setup() const noexcept { return impl_->is_setup(); }

} // namespace kopt
