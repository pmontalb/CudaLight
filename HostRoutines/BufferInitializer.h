#pragma once

#include <Flags.h>
#include <Types.h>

namespace cl { namespace routines {
	
	extern void Zero(MemoryBuffer &buf);

	extern void Initialize(MemoryBuffer &buf, const double value);

	extern void Reciprocal(MemoryBuffer &buf);

	extern void LinSpace(MemoryBuffer &buf, const double x0, const double x1);

	extern void RandUniform(MemoryBuffer &buf, const unsigned seed);

	extern void RandNormal(MemoryBuffer &buf, const unsigned seed);

	extern void Eye(MemoryTile &buf);

	extern void OnesUpperTriangular(MemoryTile &buf);

	extern void RandShuffle(MemoryBuffer &buf, const unsigned seed);

	extern void RandShufflePair(MemoryBuffer &buf1, MemoryBuffer &bu2, const unsigned seed);

	extern void RandShuffleColumns(MemoryTile &buf, const unsigned seed);

	extern void RandShuffleColumnsPair(MemoryTile &buf1, MemoryTile &bu2, const unsigned seed);
}}