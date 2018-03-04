#pragma once

#include <iostream>
#include <Types.h>

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size)
		: buffer(MemoryBuffer(0, size, ms, md)), IBuffer(true)
	{
		ctor(buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size, const double value)
		: Vector(size)
	{
		dm::detail::Initialize(buffer, value);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const Vector& rhs)
		: Vector(rhs.size())
	{		
		dm::detail::AutoCopy(buffer, rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<double> Vector<ms, md>::Get() const
	{
		dm::detail::ThreadSynchronize();

		MemoryBuffer newBuf(buffer);
		newBuf.memorySpace = MemorySpace::Host;

		dm::detail::AllocHost(newBuf);
		dm::detail::AutoCopy(newBuf, buffer);

		std::vector<double> ret(buffer.size, -123);
		detail::Fill(ret, newBuf);

		dm::detail::FreeHost(newBuf);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::Print(const std::string& label) const
	{
		auto vec = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t i = 0; i < vec.size(); i++)
			std::cout << "\tv[" << i << "] \t=\t " << vec[i] << std::endl;
		std::cout << "**********************" << std::endl;
	}

	template<MemorySpace ms, MathDomain md>
	template<MemorySpace msRhs, MathDomain mdRhs>
	bool Vector<ms, md>::operator==(const Vector<msRhs, mdRhs>& rhs) const
	{
		const auto& thisVector = Get();
		const auto& thatVector = rhs.Get();

		if (thisVector.size() != thatVector.size())
			return false;
		constexpr double tolerance = GetTolerance();
		for (size_t i = 0; i < thisVector.size(); ++i)
		{
			if (fabs(thisVector[i] - thatVector[i]) > tolerance)
				return false;
		}

		return true;
	}

	#pragma region Linear Algebra

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::operator +(const Vector& rhs) const
	{
		Vector ret(*this);
		ret += rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::operator -(const Vector& rhs) const
	{
		Vector ret(*this);
		ret -= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::operator %(const Vector& rhs) const
	{
		Vector ret(*this);
		ret %= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::Add(const Vector& rhs, const double alpha) const
	{
		Vector ret(*this);
		ret.AddEqual(rhs, alpha);

		return ret;
	}

	#pragma endregion

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Copy(const Vector<ms, md>& source)
	{
		Vector<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> LinSpace(const double x0, const double x1, const unsigned size)
	{
		Vector<ms, md> ret(size);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> RandomUniform(const unsigned size, const unsigned seed)
	{
		Vector<ms, md> ret(size);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> RandomGaussian(const unsigned size, const unsigned seed)
	{
		Vector<ms, md> ret(size);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void Print(const Vector<ms, md>& vec, const std::string& label)
	{
		vec.Print(label);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Add(const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Scale(Vector<ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}
}
