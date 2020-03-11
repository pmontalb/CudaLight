#pragma once

#include <type_traits>

#include <Types.h>
#include <Exception.h>

#include <Vector.h>

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size)
		: IBuffer<Vector<ms, md>, ms, md>(true), _buffer(MemoryBuffer(0, size, ms, md))
	{
		this->ctor(_buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const unsigned size, const typename Traits<md>::stdType value)
		: Vector(size)
	{
		dm::detail::Initialize(_buffer, static_cast<double>(value));
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const Vector& rhs)
		: Vector(rhs.size())
	{		
		this->ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const Vector& rhs, const size_t start, const size_t end) noexcept
		: IBuffer<Vector<ms, md>, ms, md>(false),  // this is a no-copy operation, this instance doesn't own the original memory!
		  _buffer(0, static_cast<unsigned>(end - start), ms, md)
	{
		assert(end > start);
		assert(end <= rhs.size());
		_buffer.pointer = rhs._buffer.pointer + start * rhs._buffer.ElementarySize();
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(Vector&& rhs) noexcept
		: IBuffer<Vector<ms, md>, ms, md>(std::move(rhs)), _buffer(rhs._buffer)
	{
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const std::vector<typename Traits<md>::stdType>& rhs)
		: Vector(static_cast<unsigned>(rhs.size()))
	{
        this->ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const std::string& fileName, bool useMemoryMapping)
	{
		std::vector<typename Traits<md>::stdType> v;
		cl::VectorFromBinaryFile(v, fileName, useMemoryMapping);

        this->ReadFrom(v);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>::Vector(const MemoryBuffer& buffer)
		: IBuffer<Vector<ms, md>, ms, md>(false), _buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::RandomShuffle(const unsigned seed)
	{
		assert(_buffer.pointer != 0);
		dm::detail::RandShuffle(_buffer, seed);
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::Print(const std::string& label) const
	{
		auto v = this->Get();
		cl::Print(v, label);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& Vector<ms, md>::ToOutputStream(std::ostream& os) const
	{
		cl::VectorToOutputStream(this->Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::ToBinaryFile(const std::string& fileName, const bool compressed, const std::string mode) const
	{
		cl::VectorToBinaryFile(this->Get(), fileName, compressed, mode);
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
	Vector<ms, md> Vector<ms, md>::Copy(const Vector<ms, md>& source)
	{
		Vector<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned size)
	{
		Vector<ms, md> ret(size);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::RandomUniform(const unsigned size, const unsigned seed)
	{
		Vector<ms, md> ret(size);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::RandomGaussian(const unsigned size, const unsigned seed)
	{
		Vector<ms, md> ret(size);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::RandomShuffle(Vector<ms, md>& v, const unsigned seed)
	{
		v.RandomShuffle(seed);
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::RandomShufflePair(Vector<ms, md>& v1, Vector<ms, md>& v2, const unsigned seed)
	{
		dm::detail::RandShufflePair(v1.GetBuffer(), v2.GetBuffer(), seed);
	}
	
	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::Print(const Vector<ms, md>& v, const std::string& label)
	{
		v.Print(label);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& Vector<ms, md>::VectorToOutputStream(const Vector<ms, md>& v, std::ostream& os)
	{
		os << cl::VectorToOutputStream(v.Get(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::VectorToBinaryFile(const Vector<ms, md>& v, const std::string& fileName, const bool compressed, const std::string mode)
	{
		const auto& _vec = v.Get();
		cl::VectorToBinaryFile(_vec, fileName, compressed, mode);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::VectorFromInputStream(std::istream& is)
	{
		std::vector<typename Vector<ms, md>::stdType> _vec;
		cl::VectorFromInputStream(_vec, is);

		Vector<ms, md> ret(static_cast<unsigned>(_vec.size()));
		ret.ReadFrom(_vec);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::VectorFromBinaryFile(const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		std::vector<typename Vector<ms, md>::stdType> _vec {};
		cl::VectorFromBinaryFile(_vec, fileName, compressed, useMemoryMapping);

		Vector<ms, md> ret(static_cast<unsigned>(_vec.size()));
		ret.ReadFrom(_vec);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Vector<ms, md>::Add(const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::Scale(Vector<ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, MathDomain::Float> Vector<ms, md>::MakePair(const Vector<ms, md>& x, const Vector<ms, md>& y)
	{
		assert(x.size() == y.size());
		Vector<ms, MathDomain::Float> pair(2 * x.size());
		MakePair(pair, x, y);

		return pair;
	}

	template<MemorySpace ms, MathDomain md>
	void Vector<ms, md>::MakePair(Vector<ms, MathDomain::Float>& pair, const Vector<ms, md>& x, const Vector<ms, md>& y)
	{
		dm::detail::MakePair(pair.GetBuffer(), x.GetBuffer(), y.GetBuffer());
	}
}
