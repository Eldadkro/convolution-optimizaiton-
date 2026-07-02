# Kernel Optimization

Minimal CMake project for benchmarking a 5x5 convolution on 1024x2048 images.

## Components

- `kopt_cpu`: native C++ 5x5 convolution.
- `kopt_cuda`: CUDA 5x5 convolution entry point and kernel.
- `kopt_bench`: command-line benchmark that loads a binary PGM image and runs N convolutions.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To build without CUDA:

```bash
cmake -S . -B build -DKOPT_BUILD_CUDA=OFF
cmake --build build -j
```

## Run

```bash
./build/kopt_bench data/image.pgm 100 cpu
./build/kopt_bench data/image.pgm 100 cuda
```

The benchmark expects a binary PGM (`P5`) image. It warns when the image is not 2048x1024.
