include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLight)

include(cmake/All.cmake)
if (CONFIG_FOUND_CUDA_GPU)
    set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)
endif()

# Npy++
add_subdirectory(../NpyCpp ${CMAKE_BINARY_DIR}/Npy++ EXCLUDE_FROM_ALL)

# Cuda Wrappers
set(CUDA_KERNEL_INCLUDE "")
set(CUDA_KERNEL_DEPENDENCY "")
set(CUDA_KERNEL_TEST_SOURCE "")
mark_as_advanced(CUDA_KERNEL_INCLUDE)
mark_as_advanced(CUDA_KERNEL_DEPENDENCIES)
mark_as_advanced(CUDA_KERNEL_TEST_SOURCE)
if (CONFIG_FOUND_CUDA_GPU)
    # CudaLightKernels
    add_subdirectory(../CudaLightKernels ${CMAKE_BINARY_DIR}/CudaLightKernels EXCLUDE_FROM_ALL)
    set(CUDA_KERNEL_INCLUDE ../CudaLightKernels)
    set(CUDA_KERNEL_DEPENDENCY CudaLightKernels)
    set(CUDA_KERNEL_TEST_SOURCE     UnitTests/CuBlasTests.cpp
                                    UnitTests/CubTests.cpp
                                    UnitTests/CuSparseTests.cpp
                                    UnitTests/DeviceManagerTests.cpp
                                    UnitTests/MatrixTests.cpp
                                    UnitTests/SerializationTests.cpp
                                    UnitTests/SparseMatrixTests.cpp
                                    UnitTests/SparseVectorTests.cpp
                                    UnitTests/TensorTests.cpp
                                    UnitTests/VectorTests.cpp)
endif()

# Mkl Wrappers
set(MKL_WRAPPERS_SOURCES "")
set(MKL_WRAPPERS_DEFINE "")
set(MKL_WRAPPERS_INCLUDE "")
set(MKL_WRAPPERS_DEPENDENCIES "")
set(MKL_TEST_SOURCE "")
mark_as_advanced(MKL_WRAPPERS_SOURCES)
mark_as_advanced(MKL_WRAPPERS_DEFINE)
mark_as_advanced(MKL_WRAPPERS_INCLUDE)
if (MKL_FOUND)
    set(MKL_WRAPPERS_SOURCES HostRoutines/MklWrappers.cpp)
    set(MKL_WRAPPERS_DEFINE USE_MKL)
    set(MKL_WRAPPERS_INCLUDE ${MKL_INCLUDE_DIR})
    set(MKL_WRAPPERS_DEPENDENCIES ${MKL_LIBRARIES})
    set(MKL_TEST_SOURCE     UnitTests/MklVectorTests.cpp)
endif()

create_library(
    NAME
        HostRoutines
    SOURCES
        HostRoutines/BufferInitializer.cpp
        HostRoutines/MemoryManager.cpp

        HostRoutines/BlasWrappers.cpp
        HostRoutines/Extra.cpp

        ${MKL_WRAPPERS_SOURCES}

    PUBLIC_INCLUDE_DIRECTORIES
        . HostRoutines ${CUDA_KERNEL_INCLUDE}
    PUBLIC_SYSTEM_INCLUDE_DIRECTORIES
        ${MKL_WRAPPERS_INCLUDE}
    PUBLIC_COMPILE_DEFINITIONS
        ${MKL_WRAPPERS_DEFINE}
    DEPENDENCIES
        ${MKL_WRAPPERS_DEPENDENCIES}
)

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
        Npy++ ${CUDA_KERNEL_DEPENDENCY} HostRoutines
)

find_package(GTest REQUIRED)
create_test(
    NAME
        CLTests
    SOURCES
        UnitTests/main.cpp

        ${CUDA_KERNEL_TEST_SOURCE}

        ${MKL_TEST_SOURCE}

        UnitTests/HostVectorTests.cpp
        UnitTests/HostBlasTests.cpp
        UnitTests/HostExtraRoutinesTests.cpp
        UnitTests/HostSerializationTests.cpp
    PUBLIC_INCLUDE_DIRECTORIES
        ${GTEST_INCLUDE_DIR}
    DEPENDENCIES
        CudaLight
    SYSTEM_DEPENDENCIES
        gtest pthread
)
