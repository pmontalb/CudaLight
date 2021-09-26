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

#define GENERIC_API_NAMESPACE oblas
#define GENERIC_API_ROUTINES_NAMESPACE obr

#ifdef USE_OPEN_BLAS
	#define GENERIC_API_DEFINE
#endif
#include <GenericBlasApiBufferInitializer.h>

#define GENERIC_API_NAMESPACE oblas
#define GENERIC_API_ROUTINES_NAMESPACE obr

#ifdef USE_OPEN_BLAS
	#define GENERIC_API_DEFINE
#endif
#include <GenericBlasApiWrappers.h>
