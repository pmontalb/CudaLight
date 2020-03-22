#pragma once

#include <CudaLight/Traits.h>

namespace cl { namespace routines {

	template<MathDomain md>
	static inline typename Traits<md>::stdType* GetPointer(const MemoryBuffer& buffer) { return reinterpret_cast<typename Traits<md>::stdType*>(buffer.pointer); }

	template<MathDomain md>
	static inline typename Traits<md>::stdType** GetRefPointer(MemoryBuffer& buffer) { return reinterpret_cast<typename Traits<md>::stdType**>(&buffer.pointer); }
}}
