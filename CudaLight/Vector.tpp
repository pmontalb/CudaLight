#pragma once

#include <type_traits>

#include <Types.h>
#include <Exception.h>

#include <Vector.h>

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size)
		: IBuffer(true), buffer(MemoryBuffer(0, size, ms, md))
	{
		ctor(buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size, const typename Traits<md>::stdType value)
		: Vector(size)
	{
		dm::detail::Initialize(buffer, value);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const Vector& rhs)
		: Vector(rhs.size())
	{		
		ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const std::vector<typename Traits<md>::stdType>& rhs)
		: Vector(static_cast<unsigned>(rhs.size()))
	{
		ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const std::string& fileName, bool useMemoryMapping)
	{
		std::vector<typename Traits<md>::stdType> vec;
		cl.VectorFromBinaryFile(vec, fileName, useMemoryMapping);

		ReadFrom(vec);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const MemoryBuffer& buffer)
		: IBuffer(false), buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::Print(const std::string& label) const
	{
		auto vec = Get();
		cl::Print(vec);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& Vector<ms, md>::ToOutputStream(std::ostream& os) const
	{
		cl::VectorToOutputStream(Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::ToBinaryFile(const std::string& fileName, const std::string mode) const
	{
		cl::VectorToBinaryFile(Get(), fileName, mode);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const Vector<ms, md>& buffer)
	{
		buffer.ToOutputStream(os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	std::istream& operator>>(std::istream& is, const Vector<ms, md>& buffer)
	{
		return buffer.Deserialize(is);
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
	Vector<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned size)
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
	std::ostream& VectorToOutputStream(const Vector<ms, md>& vec, std::ostream& os)
	{
		os << cl::VectorToOutputStream(vec.Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void VectorToBinaryFile(const Vector<ms, md>& vec, const std::string& fileName, const std::string mode)
	{
		const auto& _vec = vec.Get();
		cl::VectorToBinaryFile(_vec, fileName, mode);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> VectorFromInputStream(std::istream& is)
	{
		std::vector<Vector<ms, md>::stdType> _vec;
		cl::VectorFromInputStream(_vec, is);

		Vector<ms, md> ret(static_cast<unsigned>(_vec.size()));
		ret.ReadFrom(_vec);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> VectorFromBinaryFile(const std::string& fileName, const bool useMemoryMapping)
	{
		std::vector<Vector<ms, md>::stdType> _vec;
		cl::VectorFromBinaryFile(_vec, fileName, useMemoryMapping);

		Vector<ms, md> ret(static_cast<unsigned>(_vec.size()));
		ret.ReadFrom(_vec);

		return ret;
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
