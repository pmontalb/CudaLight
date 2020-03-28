include_guard()

cmake_minimum_required(VERSION 3.14)
project(CudaLight)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
if (CONFIG_FOUND_CUDA_GPU)
    set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)
endif()

set(LINEAR_ALGEBRA_PROVIDER "MKL" CACHE STRING "Linear algebra provider")
set_property(CACHE LINEAR_ALGEBRA_PROVIDER PROPERTY STRINGS MKL OpenBlas Blas)

# Npy++ submodule
add_subdirectory(NpyCpp ${CMAKE_BINARY_DIR}/CudaLight/NpyCpp EXCLUDE_FROM_ALL)

# CudaLightKernels submodule
add_subdirectory(CudaLightKernels ${CMAKE_BINARY_DIR}/CudaLight/CudaLightKernels EXCLUDE_FROM_ALL)

# Cuda Wrappers
set(CUDA_KERNEL_INCLUDE "")
set(CUDA_KERNEL_DEPENDENCY "")
set(CUDA_KERNEL_TEST_SOURCE "")
mark_as_advanced(CUDA_KERNEL_INCLUDE)
mark_as_advanced(CUDA_KERNEL_DEPENDENCIES)
mark_as_advanced(CUDA_KERNEL_TEST_SOURCE)
if (CONFIG_FOUND_CUDA_GPU)
    set(CUDA_KERNEL_INCLUDE CudaLightKernels)
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
set(MKL_WRAPPERS_DEFINE "")
set(MKL_WRAPPERS_DEPENDENCIES "")
set(MKL_TEST_SOURCE "")
mark_as_advanced(MKL_WRAPPERS_DEFINE)
mark_as_advanced(MKL_WRAPPERS_DEPENDENCIES "")
mark_as_advanced(MKL_TEST_SOURCE "")

# OpenBLAS Wrappers
set(OBLAS_WRAPPERS_DEFINE "")
set(OBLAS_WRAPPERS_DEPENDENCIES "")
set(OBLAS_TEST_SOURCE "")
mark_as_advanced(OBLAS_WRAPPERS_DEFINE)
mark_as_advanced(OBLAS_WRAPPERS_DEPENDENCIES "")
mark_as_advanced(OBLAS_TEST_SOURCE "")

# Generic BLAS Wrappers
set(GBLAS_WRAPPERS_DEFINE "")
set(GBLAS_WRAPPERS_DEPENDENCIES "")
set(GBLAS_TEST_SOURCE "")
mark_as_advanced(GBLAS_WRAPPERS_DEFINE)
mark_as_advanced(GBLAS_WRAPPERS_DEPENDENCIES "")
mark_as_advanced(GBLAS_TEST_SOURCE "")

if (LINEAR_ALGEBRA_PROVIDER STREQUAL "MKL" AND MKL_FOUND)
	message(STATUS "Using MKL provider")
    set(MKL_WRAPPERS_DEFINE USE_MKL)
    set(MKL_WRAPPERS_DEPENDENCIES MklRoutines)
    set(MKL_TEST_SOURCE     UnitTests/MklVectorTests.cpp
                            UnitTests/MklMatrixTests.cpp
                            UnitTests/MklTensorTests.cpp
                            UnitTests/MklBlasTests.cpp
                            UnitTests/MklSparseVectorTests.cpp
                            UnitTests/MklSparseMatrixTests.cpp
                            UnitTests/MklSparseTests.cpp
                            UnitTests/MklExtraRoutinesTests.cpp)
	create_library(
		NAME
			MklRoutines
		SOURCES
			HostRoutines/MklAllWrappers.cpp
		PUBLIC_INCLUDE_DIRECTORIES
			. HostRoutines ${CUDA_KERNEL_INCLUDE}
		PUBLIC_SYSTEM_INCLUDE_DIRECTORIES
			${MKL_INCLUDE_DIR} CudaLightKernels
		PUBLIC_COMPILE_DEFINITIONS
			${MKL_WRAPPERS_DEFINE}
		DEPENDENCIES
			${MKL_LIBRARIES}
	)
elseif (LINEAR_ALGEBRA_PROVIDER STREQUAL "OpenBlas" AND OpenBLAS_FOUND)
	message(STATUS "Using OpenBlas provider")

	set(OBLAS_WRAPPERS_DEFINE USE_OPEN_BLAS)
	set(OBLAS_WRAPPERS_DEPENDENCIES OpenBlasRoutines)
	set(OBLAS_TEST_SOURCE   UnitTests/OpenBlasVectorTests.cpp
							UnitTests/OpenBlasMatrixTests.cpp
							UnitTests/OpenBlasTensorTests.cpp
							UnitTests/OpenBlasTests.cpp
							UnitTests/OpenBlasExtraRoutinesTests.cpp)
	create_library(
		NAME
			OpenBlasRoutines
		SOURCES
			HostRoutines/OpenBlasAllWrappers.cpp
		PUBLIC_INCLUDE_DIRECTORIES
			. HostRoutines ${CUDA_KERNEL_INCLUDE}
		PUBLIC_SYSTEM_INCLUDE_DIRECTORIES
			${OpenBLAS_INCLUDE_DIR} CudaLightKernels
		PUBLIC_COMPILE_DEFINITIONS
			${OBLAS_WRAPPERS_DEFINE}
		DEPENDENCIES
			${OpenBLAS_LIB} ${LAPACK_LIBRARIES}
	)
elseif (LINEAR_ALGEBRA_PROVIDER STREQUAL "Blas" AND BLAS_FOUND AND LAPACK_FOUND)
	message(STATUS "Using generic Blas provider")

	set(GBLAS_WRAPPERS_DEFINE USE_BLAS)
	set(GBLAS_WRAPPERS_DEPENDENCIES GenericBlasRoutines)
	set(GBLAS_TEST_SOURCE   UnitTests/GenericBlasVectorTests.cpp
							UnitTests/GenericBlasMatrixTests.cpp
							UnitTests/GenericBlasTensorTests.cpp
							UnitTests/GenericBlasTests.cpp
							UnitTests/GenericBlasExtraRoutinesTests.cpp)
	create_library(
		NAME
			GenericBlasRoutines
		SOURCES
			HostRoutines/GenericBlasAllWrappers.cpp
		PUBLIC_INCLUDE_DIRECTORIES
			. HostRoutines ${CUDA_KERNEL_INCLUDE}
		PUBLIC_COMPILE_DEFINITIONS
			${GBLAS_WRAPPERS_DEFINE} CudaLightKernels
		DEPENDENCIES
			${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
	)
endif()

create_library(
    NAME
        HostRoutines
    SOURCES
        HostRoutines/BufferInitializer.cpp
        HostRoutines/MemoryManager.cpp
        HostRoutines/BlasWrappers.cpp
        HostRoutines/SparseWrappers.cpp
        HostRoutines/Extra.cpp
        HostRoutines/ForgeHelpers.cpp
    PUBLIC_INCLUDE_DIRECTORIES
        . HostRoutines CudaLightKernels ${CUDA_KERNEL_INCLUDE}
    DEPENDENCIES
        ${MKL_WRAPPERS_DEPENDENCIES} ${OBLAS_WRAPPERS_DEPENDENCIES} ${GBLAS_WRAPPERS_DEPENDENCIES}
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

# gtest
#add_subdirectory(gtest ${CMAKE_BINARY_DIR}/CudaLight/gtest EXCLUDE_FROM_ALL)
create_test(
    NAME
        CLTests
    SOURCES
        UnitTests/main.cpp

        ${CUDA_KERNEL_TEST_SOURCE}

        ${MKL_TEST_SOURCE}
		${OBLAS_TEST_SOURCE}
		${GBLAS_TEST_SOURCE}

        UnitTests/HostVectorTests.cpp
        UnitTests/HostMatrixTests.cpp
        UnitTests/HostTensorTests.cpp
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
