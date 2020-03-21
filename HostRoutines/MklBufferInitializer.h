#pragma once

#include <Types.h>
#include <Exceptions.h>

#ifndef USE_MKL

	namespace cl { namespace routines { namespace mkr {

		template<MathDomain md>
		void Copy(MemoryBuffer& dest, const MemoryBuffer& source);

		template<>
		inline void Copy<MathDomain::Float>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			throw NotImplementedException();
		}
		template<>
		inline void Copy<MathDomain::Double>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void RandUniform(MemoryBuffer& buf, const unsigned seed = 1234);

		template<>
		inline void RandUniform<MathDomain::Float>(MemoryBuffer& buf, const unsigned seed)
		{
			throw NotImplementedException();
		}
		template<>
		inline void RandUniform<MathDomain::Double>(MemoryBuffer& buf, const unsigned seed)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void RandNormal(MemoryBuffer& buf, const unsigned seed = 1234);

		template<>
		inline void RandNormal<MathDomain::Double>(MemoryBuffer& buf, const unsigned seed)
		{
			throw NotImplementedException();
		}

		template<>
		inline void RandNormal<MathDomain::Float>(MemoryBuffer& buf, const unsigned seed)
		{
			throw NotImplementedException();
		}
	}}}

#else

	namespace mkl
	{
		#include <mkl.h>
	}

	namespace cl { namespace routines { namespace mkr {

		template<MathDomain md>
		void Copy(MemoryBuffer& dest, const MemoryBuffer& source);

		template<>
		inline void Copy<MathDomain::Float>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			static constexpr int incx = 1;
			static constexpr int incy = 1;
			const auto sz = static_cast<int>(dest.size);
			mkl::scopy(&sz, reinterpret_cast<const float*>(source.pointer), &incx, reinterpret_cast<float*>(dest.pointer), &incy);
		}
		template<>
		inline void Copy<MathDomain::Double>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			static constexpr int incx = 1;
			static constexpr int incy = 1;
			const auto sz = static_cast<int>(dest.size);
			mkl::dcopy(&sz, reinterpret_cast<const double*>(source.pointer), &incx, reinterpret_cast<double*>(dest.pointer), &incy);
		}

		template<MathDomain md>
		static void RandUniform(MemoryBuffer& buf, const unsigned seed = 1234);

		template<>
		inline void RandUniform<MathDomain::Float>(MemoryBuffer& buf, const unsigned seed)
		{
			mkl::VSLStreamStatePtr stream;
			mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
			auto err = mkl::vsRngUniform(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, static_cast<int>(buf.size), reinterpret_cast<float*>(buf.pointer), 0.0f, 1.0f);
			if (err != 0)
				throw MklException(__func__);
			mkl::vslDeleteStream(&stream);
		}
		template<>
		inline void RandUniform<MathDomain::Double>(MemoryBuffer& buf, const unsigned seed)
		{
			mkl::VSLStreamStatePtr stream;
			mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
			auto err = mkl::vdRngUniform(VSL_RNG_METHOD_UNIFORMBITS_STD, stream, static_cast<int>(buf.size), reinterpret_cast<double*>(buf.pointer), 0.0, 1.0);
			if (err != 0)
				throw MklException(__func__);
			mkl::vslDeleteStream(&stream);
		}

		template<MathDomain md>
		static void RandNormal(MemoryBuffer& buf, const unsigned seed = 1234);

		template<>
		inline void RandNormal<MathDomain::Float>(MemoryBuffer& buf, const unsigned seed)
		{
			mkl::VSLStreamStatePtr stream;
			mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
			auto err = mkl::vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, static_cast<int>(buf.size), reinterpret_cast<float*>(buf.pointer), 0.0, 1.0);
			if (err != 0)
				throw MklException(__func__);
			mkl::vslDeleteStream(&stream);
		}

		template<>
		inline void RandNormal<MathDomain::Double>(MemoryBuffer& buf, const unsigned seed)
		{
			mkl::VSLStreamStatePtr stream;
			mkl::vslNewStream(&stream, VSL_BRNG_MT19937, seed);
			auto err = mkl::vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, static_cast<int>(buf.size), reinterpret_cast<double*>(buf.pointer), 0.0, 1.0);
			if (err != 0)
				throw MklException(__func__);
			mkl::vslDeleteStream(&stream);
		}
	}}}

#endif