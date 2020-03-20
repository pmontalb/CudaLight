#pragma once

#include <Types.h>

#ifndef USE_MKL

	// TODO: keep the same API

#else

namespace mkl
{
	#include <mkl.h>
}


namespace cl { namespace routines { namespace mkr {
	static inline void Copy(float* __restrict__ dest, const float* __restrict__ source, const int size)
	{
		int incx = 1;
		int incy = 1;
		mkl::scopy(&size, source, &incx, dest, &incy);
	}
	static inline void Copy(double* __restrict__ dest, const double* __restrict__ source, const int size)
	{
		int incx = 1;
		int incy = 1;
		mkl::dcopy(&size, source, &incx, dest, &incy);
	}

	template<typename T>
	static inline void Alloc(T** ptr, const size_t size)
	{
		static constexpr int alignmentBits = { 64 };
		*ptr = reinterpret_cast<T*>(mkl::MKL_malloc(size * sizeof(T), alignmentBits));
		assert(*ptr != nullptr);
	}

	template<typename T>
	static inline void Free(T** ptr)
	{
		mkl::MKL_free(*ptr);
	}

	static inline void RandUniform(float* __restrict__ x, const size_t size, const unsigned seed = 1234)
	{
		mkl::VSLStreamStatePtr stream;
		mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
		mkl::vsRngUniform(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, static_cast<int>(size), x, 0.0f, 1.0f);
		mkl::vslDeleteStream(&stream);
	}

	static inline void RandUniform(double* __restrict__ x, const size_t size, const unsigned seed = 1234)
	{
		mkl::VSLStreamStatePtr stream;
		mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
		mkl::vdRngUniform(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, static_cast<int>(size), x, 0.0, 1.0);
		mkl::vslDeleteStream(&stream);
	}

	static inline void RandNormal(float* __restrict__ x, const size_t size, const unsigned seed = 1234)
	{
		mkl::VSLStreamStatePtr stream;
		mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
		mkl::vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, static_cast<int>(size), x, 0.0, 1.0);
		mkl::vslDeleteStream(&stream);
	}

	static inline void RandNormal(double* __restrict__ x, const size_t size, const unsigned seed = 1234)
	{
		mkl::VSLStreamStatePtr stream;
		mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
		mkl::vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, static_cast<int>(size), x, 0.0, 1.0);
		mkl::vslDeleteStream(&stream);
	}

	/**
	* z = alpha * x + y
	*/
	static inline void Add(float* __restrict__ z, const float* __restrict__ x, const float* __restrict__ y, const int size, const float alpha = 1.0f)
	{
		Copy(z, y, size);

		int incx = 1;
		int incy = 1;
		mkl::saxpy(&size, &alpha, x, &incx, z, &incy);
	}
	static inline void Add(double* __restrict__ z, const double* __restrict__ x, const double* __restrict__ y, const int size, const double alpha = 1.0)
	{
		Copy(z, y, size);

		int incx = 1;
		int incy = 1;
		mkl::daxpy(&size, &alpha, x, &incx, z, &incy);
	}









	// norm = ||x||_2
	static inline void EuclideanNorm(float& norm, const float* __restrict__ x, const int size)
	{
		int incx = 1;
		norm = mkl::snrm2(&size, x, &incx);
	}
	static inline void EuclideanNorm(double& norm, const double* __restrict__ x, const int size)
	{
		int incx = 1;
		norm = mkl::dnrm2(&size, x, &incx);
	}
}}}

#endif