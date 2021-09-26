
#include <Exceptions.h>
#include <Types.h>

#ifndef GENERIC_API_NAMESPACE
	#error "Wrong usage of this header"
#endif

#ifndef GENERIC_API_ROUTINES_NAMESPACE
	#error "Wrong usage of this header"
#endif

#define ROUTINES_NAMESPACE namespace GENERIC_API_ROUTINES_NAMESPACE
#define BLAS_NAMESPACE namespace GENERIC_API_NAMESPACE

#ifndef GENERIC_API_DEFINE

namespace cl
{
	namespace routines
	{
		ROUTINES_NAMESPACE
		{
			template<MathDomain md>
			void Copy(MemoryBuffer&, const MemoryBuffer&)
			{
				throw NotImplementedException();
			}
		}
	}	 // namespace routines
}	 // namespace cl

#else


BLAS_NAMESPACE
{
	#ifdef CBLAS_H
		#undef CBLAS_H
	#endif

	#include <cblas.h>
}

namespace cl
{
	namespace routines
	{
		ROUTINES_NAMESPACE
		{
			template<MathDomain md>
			void Copy(MemoryBuffer & dest, const MemoryBuffer& source);

			template<>
			inline void Copy<MathDomain::Float>(MemoryBuffer & dest, const MemoryBuffer& source)
			{
				GENERIC_API_NAMESPACE::cblas_scopy(static_cast<int>(dest.size), reinterpret_cast<const float*>(source.pointer), 1, reinterpret_cast<float*>(dest.pointer), 1);
			}
			template<>
			inline void Copy<MathDomain::Double>(MemoryBuffer & dest, const MemoryBuffer& source)
			{
				GENERIC_API_NAMESPACE::cblas_dcopy(static_cast<int>(dest.size), reinterpret_cast<const double*>(source.pointer), 1, reinterpret_cast<double*>(dest.pointer), 1);
			}
		}
	}	 // namespace routines
}	 // namespace cl

#endif

#undef GENERIC_API_DEFINE
#undef GENERIC_API_NAMESPACE
#undef GENERIC_API_ROUTINES_NAMESPACE
#undef ROUTINES_NAMESPACE
#undef BLAS_NAMESPACE
