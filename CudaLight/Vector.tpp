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
	std::ostream& Vector<ms, md>::Serialize(std::ostream& os) const
	{
		cl::SerializeVector(Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const Vector<ms, md>& buffer)
	{
		buffer.Serialize(os);
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
	std::ostream& SerializeVector(const Vector<ms, md>& vec, std::ostream& os)
	{
		os << cl::SerializeVector(vec.Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> DeserializeVector(std::istream& is)
	{
		std::vector<Vector<ms, md>::stdType> _vec;
		cl::DeserializeVector(_vec, is);

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
