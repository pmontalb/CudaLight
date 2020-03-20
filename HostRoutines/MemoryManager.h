#pragma once

#include <Flags.h>
#include <Types.h>

namespace cl { namespace routines {

	extern void Copy(MemoryBuffer& dest, const MemoryBuffer& source);

	extern void Alloc(MemoryBuffer& ptr);

	extern void Free(MemoryBuffer& ptr);
}}
