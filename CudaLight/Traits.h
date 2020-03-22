#pragma once

#include <Types.h>

namespace cl
{
	// This is required for using types defined in children classes (which are incomplete types)

	template <MathDomain mathDomain>
	struct Traits
	{
		using stdType = void;
	};

	template <>
	struct Traits<MathDomain::Double>
	{
		using stdType = double;
	};

	template <>
	struct Traits<MathDomain::Float>
	{
		using stdType = float;
	};

	template <>
	struct Traits<MathDomain::Int>
	{
		using stdType = int;
	};

	template <typename T>
	struct _Traits
	{
		static constexpr MathDomain clType = MathDomain::Null;
	};

	template <>
	struct _Traits<double>
	{
		static constexpr MathDomain clType = MathDomain::Double;
	};

	template <>
	struct _Traits<float>
	{
		static constexpr MathDomain clType = MathDomain::Float;
	};

	template <>
	struct _Traits<int>
	{
		static constexpr MathDomain clType = MathDomain::Int;
	};
}
