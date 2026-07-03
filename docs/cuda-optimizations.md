# CUDA Optimization Summary

This document records the CUDA-side optimization steps applied to the 5x5 LPF/HPF benchmark and the measured impact during local runs on `data/image.png` with 100 iterations.

Test system:

| Item | Value |
| --- | --- |
| CPU | 13th Gen Intel Core i9-13900K |
| CPU SIMD target | `-O3 -march=native` |
| CUDA build | `build-release-cuda` |
| Input | `data/image.png` |
| Benchmark command | `build-release-cuda/kopt_bench data/image.png 100 cuda` |
| OpenCV threads | 1 |

## Changes

| Step | Upgrade | What Changed | Why It Helped | Measured CUDA Avg |
| --- | --- | --- | --- | --- |
| 1 | Baseline CUDA convolution | The CUDA path allocated device input/output buffers and copied the 5x5 kernel every `run()` call. | Correct GPU implementation, but per-frame allocation and setup dominated runtime. | ~14.9 ms |
| 2 | Persistent CUDA state | Added `CudaConvolutionState` to own CUDA buffers across runs. Allocation and kernel constant upload moved to setup. | Removed repeated `cudaMalloc`, `cudaFree`, and constant-memory setup from the timed loop. | Not isolated |
| 3 | Direct U8 device input | CUDA state now owns a `uint8_t` device input buffer. `CudaImpl` copies the source image directly to device memory instead of first converting to a host float buffer. | Removed host-side U8-to-float conversion before every CUDA run and reduced input transfer size. | Part of ~0.78 ms |
| 4 | GPU LPF + HPF output | Added CUDA kernels to compute LPF and then HPF/output formatting on the GPU. | Removed host-side LPF/HPF output generation from the timed loop. | ~0.78 ms |
| 5 | State member API | Moved CUDA operations onto `CudaConvolutionState`: `setup`, `copy_input`, `run`, `copy_output`, `release`. | Centralized CUDA resource ownership and made ownership/lifetime explicit. | No direct timing change |
| 6 | NPP box filter wrapper | Added `CudaConvolutionState::run_npp_box_filter()` around `nppiFilterBoxBorder_8u_C1R` with replicated borders. | Uses NVIDIA's optimized NPP image filtering implementation for the 5x5 box LPF. | Wrapper only |
| 7 | NPP in active CUDA run | `CudaConvolutionState::run()` now calls NPP for LPF, then launches a CUDA kernel for HPF. | Replaces the custom 5x5 LPF kernel with a tuned library primitive. | ~0.71 ms |

## Current CUDA Pipeline

| Phase | Implementation |
| --- | --- |
| Setup | `CudaConvolutionState::setup(...)` allocates device input, LPF output, HPF output, and scratch buffers. |
| Input upload | `CudaConvolutionState::copy_input(...)` copies the original U8 image to the device. |
| LPF | `CudaConvolutionState::run_npp_box_filter()` calls `nppiFilterBoxBorder_8u_C1R`. |
| HPF | `write_filter_outputs_kernel` computes `input - lpf + 0.5` on the GPU. |
| Output download | `CudaConvolutionState::copy_output(...)` copies U8 LPF and HPF images back to host memory. |
| Cleanup | `CudaConvolutionState::~CudaConvolutionState()` calls `release()`. |

## Result

| Version | Avg Time |
| --- | ---: |
| Original CUDA path with per-run allocation | ~14.9 ms |
| Persistent state + custom CUDA U8 pipeline | ~0.78 ms |
| Persistent state + NPP box filter | ~0.71 ms |

The largest win came from moving allocation/setup out of the per-frame path and avoiding host-side conversion/output work. NPP provided an additional smaller improvement for the 5x5 box LPF.
