cmake_minimum_required(VERSION 3.18)
project(ParallelizedImageAndMatrixEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -Wall -Wextra)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling OpenMP parallelization flags")
endif()

# Find OpenCV (Strictly for Image I/O)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_definitions(-DHPC_USE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(STATUS "OpenCV not found: using built-in binary Netpbm (PGM) I/O fallback")
endif()

# Project Include directories
include_directories(include)

# Source files
set(SOURCES
    src/cpu/filter_cpu.cpp
    src/cpu/matmul_cpu.cpp
    src/io/image_io.cpp
    src/cuda/filter_cuda.cu
    src/cuda/matmul_cuda.cu
    src/main.cpp
)

# Main Executable
add_executable(hpc_engine ${SOURCES})

# Target properties for CUDA
set_target_properties(hpc_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link Libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(hpc_engine PRIVATE OpenMP::OpenMP_CXX)
endif()

if(OpenCV_FOUND)
    target_link_libraries(hpc_engine PRIVATE ${OpenCV_LIBS})
endif()
