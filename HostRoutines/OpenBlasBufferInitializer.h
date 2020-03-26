#pragma once

#include <Types.h>
#include <Exceptions.h>

#ifndef USE_OPEN_BLAS

	namespace cl { namespace routines { namespace obr {

		template<MathDomain md>
		void Copy(MemoryBuffer&, const MemoryBuffer&)
		{
			throw NotImplementedException();
		}

	}}}

#else

	namespace oblas
	{
		#include <cblas.h>
	}

	namespace cl { namespace routines { namespace obr {

		template<MathDomain md>
		void Copy(MemoryBuffer& dest, const MemoryBuffer& source);

		template<>
		inline void Copy<MathDomain::Float>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			oblas::cblas_scopy(static_cast<int>(dest.size), reinterpret_cast<const float*>(source.pointer), 1, reinterpret_cast<float*>(dest.pointer), 1);
		}
		template<>
		inline void Copy<MathDomain::Double>(MemoryBuffer& dest, const MemoryBuffer& source)
		{
			oblas::cblas_dcopy(static_cast<int>(dest.size), reinterpret_cast<const double*>(source.pointer), 1, reinterpret_cast<double*>(dest.pointer), 1);
		}
	}}}

#endif
