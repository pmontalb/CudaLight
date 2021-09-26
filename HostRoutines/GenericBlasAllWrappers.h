#pragma once

#include <BufferInitializer.h>
#include <MemoryManager.h>
#include <Types.h>

#include <array>
#include <vector>

#ifdef lapack_complex_float
	#undef lapack_complex_float
#endif

#ifdef lapack_complex_double
	#undef lapack_complex_double
#endif

#define GENERIC_API_NAMESPACE gblas
#define GENERIC_API_ROUTINES_NAMESPACE gbr

#ifdef USE_BLAS
	#define GENERIC_API_DEFINE
#endif
#include <GenericBlasApiBufferInitializer.h>

#define GENERIC_API_NAMESPACE gblas
#define GENERIC_API_ROUTINES_NAMESPACE gbr

#ifdef USE_BLAS
	#define GENERIC_API_DEFINE
#endif

#include <GenericBlasApiWrappers.h>
