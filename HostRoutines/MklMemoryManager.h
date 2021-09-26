#pragma once

#include <Exceptions.h>
#include <Types.h>

#ifndef USE_MKL

namespace cl
{
	namespace routines
	{
		namespace mkr
		{
			static inline void Alloc(MemoryBuffer&) { throw NotImplementedException(); }

			static inline void Free(MemoryBuffer&) { throw NotImplementedException(); }
		}	 // namespace mkr
	}		 // namespace routines
}	 // namespace cl

#else

namespace mkl
{
	#include <mkl.h>
}

namespace cl
{
	namespace routines
	{
		namespace mkr
		{
			static inline void Alloc(MemoryBuffer& buf)
			{
				static constexpr int alignmentBits = { 64 };
				buf.pointer = reinterpret_cast<ptr_t>(mkl::MKL_malloc(buf.TotalSize(), alignmentBits));
				assert(buf.pointer != 0);
			}

			static inline void Free(MemoryBuffer& buf) { mkl::MKL_free(reinterpret_cast<void*>(buf.pointer)); }
		}	 // namespace mkr
	}		 // namespace routines
}	 // namespace cl

#endif
