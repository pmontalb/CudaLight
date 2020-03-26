#pragma once

#ifdef lapack_complex_float
	#undef lapack_complex_float
#endif

#ifdef lapack_complex_double
	#undef lapack_complex_double
#endif

#include <complex>
#include <MklMemoryManager.h>
#include <MklBufferInitializer.h>
#include <MklBlasWrappers.h>
#include <MklSparseWrappers.h>