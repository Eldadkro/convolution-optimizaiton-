# Kernel Optimization

Minimal CMake project for benchmarking a 5x5 convolution on 1024x2048 images.

for full detail on the work checkout [Full detail](docs/README.md)

## Components

- `kopt_cpu`: native C++ 5x5 convolution.
- `kopt_cuda`: CUDA 5x5 convolution entry point and kernel.
- `kopt_bench`: command-line benchmark that loads an image with OpenCV and runs N filters.

## Build

Prerequisites:

- CMake and Make.
- A C++17-capable compiler such as `g++`.
- NVIDIA CUDA Toolkit with `nvcc` for CUDA builds.
- OpenCV 4.6 libraries installed and discoverable in your library/search path.

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
./build/kopt_bench data/image.png 100 cpu
./build/kopt_bench data/image.png 100 cuda
```

The benchmark reads the input as an 8-bit grayscale image through OpenCV. It warns when the image is not 2048x1024.
