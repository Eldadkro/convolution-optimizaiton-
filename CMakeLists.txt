cmake_minimum_required(VERSION 3.24)

project(kernel_optimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KOPT_BUILD_CUDA "Build CUDA convolution library" ON)

add_library(kopt_release_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kopt_release_options
        INTERFACE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
    )
elseif(MSVC)
    target_compile_options(kopt_release_options
        INTERFACE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/arch:AVX512>
    )
endif()

find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgcodecs imgproc)

add_library(kopt_cpu
    src/cpu/convolution.cpp
)
target_include_directories(kopt_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(kopt_cpu
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        kopt_release_options
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
    target_link_libraries(kopt_cuda
        PUBLIC
            nppif
            nppc
    )
    set_target_properties(kopt_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
endif()

add_library(kopt_filter
    src/filter.cpp
)
target_include_directories(kopt_filter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kopt_filter
    PUBLIC
        kopt_cpu
    PRIVATE
        kopt_release_options
)
if(KOPT_BUILD_CUDA)
    target_link_libraries(kopt_filter PUBLIC kopt_cuda)
    target_compile_definitions(kopt_filter PRIVATE KOPT_WITH_CUDA=1)
    set_target_properties(kopt_filter PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

add_executable(kopt_bench
    src/app/main.cpp
    src/app/image_io.cpp
    src/app/utils.cpp
)
target_include_directories(kopt_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(kopt_bench
    PRIVATE
        kopt_filter
        kopt_release_options
        ${OpenCV_LIBS}
)

if(KOPT_BUILD_CUDA)
    target_compile_definitions(kopt_bench PRIVATE KOPT_WITH_CUDA=1)
endif()
