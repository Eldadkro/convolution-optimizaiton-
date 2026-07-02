# Repository Guidelines

## Project Structure & Module Organization

This repository is a minimal CMake project for benchmarking 5x5 convolution on 2048x1024 images.

- `include/kopt/`: public headers shared by CPU, CUDA, and app code.
- `src/cpu/`: native C++ convolution implementation.
- `src/cuda/`: CUDA convolution implementation and kernel launch code.
- `src/app/`: benchmark executable, image loading, and CLI handling.
- `data/`: local input images or small fixtures. Avoid committing large datasets.
- `build/`: generated CMake build output; do not commit it.

Tests are not scaffolded yet. Add future tests under `tests/` and wire them through CTest.

## Build, Test, and Development Commands

Configure and build with CUDA enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Configure without CUDA, useful on machines without an NVIDIA toolchain:

```bash
cmake -S . -B build -DKOPT_BUILD_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the benchmark:

```bash
./build/kopt_bench data/image.pgm 100 cpu
./build/kopt_bench data/image.pgm 100 cuda
```

Run tests once added:

```bash
ctest --test-dir build --output-on-failure
```

## Coding Style & Naming Conventions

Use C++17 and CUDA C++17. Keep indentation at 4 spaces and prefer clear, small functions over broad abstractions. Public APIs live in `include/kopt/` and use the `kopt` namespace.

Use `snake_case` for functions and variables, `PascalCase` for simple structs such as `ImageView`, and `k`-prefixed names for constants such as `kImageWidth`. Keep CUDA kernels in `.cu` files and host-only code in `.cpp` files.

## Testing Guidelines

When adding tests, prefer small deterministic inputs where expected convolution output can be checked exactly or with a tight floating-point tolerance. Name test files by component, for example `tests/cpu_convolution_test.cpp` and `tests/cuda_convolution_test.cu`.

Benchmark timings are not correctness tests. Include correctness checks before relying on performance numbers.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so no existing convention is established. Use short imperative commit subjects, for example `Add CPU convolution benchmark`.

Pull requests should include a brief description, build/test commands run, target hardware when CUDA performance changes, and any known limitations. Link related issues when available and include benchmark output for optimization work.

## Security & Configuration Tips

Do not commit generated binaries, local build directories, or large image datasets. Keep machine-specific CUDA architecture tuning in CMake options or documented build commands rather than hard-coding local assumptions.
