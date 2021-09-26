#pragma once

#include <Flags.h>
#include <Types.h>

namespace cl
{
	namespace routines
	{
		extern void MakePair(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y);

		extern void MakeTriple(MemoryBuffer& v, const MemoryBuffer& x, const MemoryBuffer& y, const MemoryBuffer& z);

		// extern void MakeRgbaJetColorMap(MemoryBuffer out, const MemoryBuffer in);
	}	 // namespace routines
}	 // namespace cl
