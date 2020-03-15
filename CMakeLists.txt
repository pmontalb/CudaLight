include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLight)

include(cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

# Npy++
add_subdirectory(../NpyCpp ${CMAKE_BINARY_DIR}/Npy++ EXCLUDE_FROM_ALL)

# CudaLightKernels
add_subdirectory(../CudaLightKernels ${CMAKE_BINARY_DIR}/CudaLightKernels EXCLUDE_FROM_ALL)

create_library(
    NAME
        CudaLight
    SOURCES
        CudaLight/DeviceManager.cpp
        CudaLight/DeviceManagerHelper.cpp
        CudaLight/Exception.cpp
    PUBLIC_INCLUDE_DIRECTORIES
        CudaLight
    DEPENDENCIES
        Npy++ CudaLightKernels
)

find_package(GTest REQUIRED)
create_test(
    NAME
        CLTests
    SOURCES
        UnitTests/main.cpp
        UnitTests/stdafx.cpp

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
    PUBLIC_INCLUDE_DIRECTORIES
        ${GTEST_INCLUDE_DIR}
    DEPENDENCIES
        CudaLight
    SYSTEM_DEPENDENCIES
        gtest pthread
)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(CLTests PRIVATE -Wno-absolute-value)
endif()
