#pragma once

#include <Types.h>

namespace cl { namespace routines
{
	extern void Sum(double& sum, const MemoryBuffer& v);

	extern void Min(double& min, const MemoryBuffer& x);

	extern void Max(double& max, const MemoryBuffer& x);

	extern void AbsMin(double& min, const MemoryBuffer& x);

	extern void AbsMax(double& max, const MemoryBuffer& x);
}}