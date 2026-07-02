#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kopt {

enum class Backend { Auto, Cpu, Cuda };

struct ImageViewU8 {
  const std::uint8_t *data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride_bytes = 0;
};

struct MutableImageViewU8 {
  std::uint8_t *data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride_bytes = 0;
};

struct FilterConfig {
  Backend backend = Backend::Auto;
  int radius = 2;
};

class FilterEngine {
public:
  FilterEngine();
  explicit FilterEngine(FilterConfig config);
  ~FilterEngine();

  FilterEngine(FilterEngine &&) noexcept;
  FilterEngine &operator=(FilterEngine &&) noexcept;

  FilterEngine(const FilterEngine &) = delete;
  FilterEngine &operator=(const FilterEngine &) = delete;

  void init(FilterConfig config = {});
  void setup(int width, int height);
  void run(ImageViewU8 input, MutableImageViewU8 lpf_output,
           MutableImageViewU8 hpf_output);

  [[nodiscard]] Backend active_backend() const noexcept;
  [[nodiscard]] bool is_initialized() const noexcept;
  [[nodiscard]] bool is_setup() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace kopt
