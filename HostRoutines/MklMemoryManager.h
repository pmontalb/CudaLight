#pragma once

#include <Types.h>
#include <Exceptions.h>

#ifndef USE_MKL

	namespace cl { namespace routines { namespace mkr
	{
		static inline void Alloc(MemoryBuffer &buf)
		{
			throw NotImplementedException();
		}

		static inline void Free(MemoryBuffer &buf)
		{
			throw NotImplementedException();
		}
	}}}

#else

	namespace mkl
	{
	#include <mkl.h>
	}

	namespace cl { namespace routines { namespace mkr
	{
		static inline void Alloc(MemoryBuffer &buf)
		{
			static constexpr int alignmentBits = {64};
			buf.pointer = reinterpret_cast<ptr_t>(mkl::MKL_malloc(buf.TotalSize(), alignmentBits));
			assert(buf.pointer != 0);
		}

		static inline void Free(MemoryBuffer &buf)
		{
			mkl::MKL_free(reinterpret_cast<void *>(buf.pointer));
		}
	}}}

#endif
