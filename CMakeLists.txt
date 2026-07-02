cmake_minimum_required(VERSION 3.24)

project(kernel_optimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KOPT_BUILD_CUDA "Build CUDA convolution library" ON)

add_library(kopt_cpu
    src/cpu/convolution.cpp
)
target_include_directories(kopt_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(KOPT_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(kopt_cuda
        src/cuda/convolution.cu
    )
    target_include_directories(kopt_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    set_target_properties(kopt_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
endif()

add_executable(kopt_bench
    src/app/main.cpp
    src/app/pgm_image.cpp
)
target_include_directories(kopt_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kopt_bench
    PRIVATE
        kopt_cpu
)

if(KOPT_BUILD_CUDA)
    target_link_libraries(kopt_bench PRIVATE kopt_cuda)
    target_compile_definitions(kopt_bench PRIVATE KOPT_WITH_CUDA=1)
endif()
