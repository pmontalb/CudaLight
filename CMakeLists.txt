cmake_minimum_required(VERSION 3.14)
project(CudaLight)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)
enable_language(CUDA)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

include(ExternalProject)
include(GNUInstallDirs)

# Npy++
include_directories(../NpyCpp/MemoryMapping)
include_directories(../NpyCpp/Npy++)
add_subdirectory(../NpyCpp ${CMAKE_BINARY_DIR}/Npy++ EXCLUDE_FROM_ALL)

# CudaLightKernels
include_directories(../CudaLightKernels)
add_subdirectory(../CudaLightKernels ${CMAKE_BINARY_DIR}/CudaLightKernels EXCLUDE_FROM_ALL)

include_directories(CudaLight)
include_directories(UnitTests)

# set warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(WARNINGS -Weverything -Werror
            -Wno-c++98-compat -Wno-c++98-compat-pedantic
            -Wno-reserved-id-macro -Wno-switch-enum -Wno-unknown-pragmas -Wno-disabled-macro-expansion
            -Wno-unneeded-internal-declaration -Wno-weak-vtables -Wno-unused-function -Wno-padded -Wno-unused-macros)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-function)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(WARNINGS -Wall -Wextra -Werror -Wno-unknown-pragmas -Wfatal-errors -fstrict-aliasing -pedantic -pedantic-errors)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(FATAL "MSVC not supported")
endif()

add_library(
        CudaLight

        CudaLight/CudaException.h

        CudaLight/DeviceManager.cpp
        CudaLight/DeviceManager.h

        CudaLight/DeviceManagerHelper.cpp
        CudaLight/DeviceManagerHelper.h

        CudaLight/Exception.cpp
        CudaLight/Exception.h

        CudaLight/IBuffer.h
        CudaLight/IBuffer.tpp

        CudaLight/Vector.h
        CudaLight/Vector.tpp

        CudaLight/SparseVector.h
        CudaLight/SparseVector.tpp

        CudaLight/ColumnWiseMatrix.h
        CudaLight/ColumnWiseMatrix.tpp

        CudaLight/CompressedSparseRowMatrix.h
        CudaLight/CompressedSparseRowMatrix.tpp

        CudaLight/Tensor.h
        CudaLight/Tensor.tpp
)
target_compile_options(CudaLight PRIVATE ${WARNINGS})
target_link_libraries(CudaLight Npy++ CudaLightKernels)

add_executable(
        CLTests

        UnitTests/main.cpp
        UnitTests/stdafx.cpp
        UnitTests/stdafx.h
        UnitTests/targetver.h

        UnitTests/CuBlasTests.cpp
        UnitTests/CubTests.cpp
        UnitTests/CuSparseTests.cpp
        UnitTests/DeviceManagerTests.cpp
        UnitTests/MatrixTests.cpp
        UnitTests/SerializationTests.cpp
        UnitTests/SparseMatrixTests.cpp
        UnitTests/SparseVectorTests.cpp
        UnitTests/TensorTests.cpp
        UnitTests/VectorTests.cpp
)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(CLTests PRIVATE -Wno-absolute-value)
endif()
target_link_libraries(CLTests CudaLight gtest pthread)
