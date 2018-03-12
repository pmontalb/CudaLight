#pragma once

#include <iostream>
#include <assert.h>

#include <Types.h>

namespace cl
{
	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>::IBuffer(const bool isOwner)
		: isOwner(isOwner)
	{
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::ctor(MemoryBuffer& buffer)
	{
		// if is not the owner it has already been allocated!
		if (!isOwner)
		{
			assert(buffer.pointer != 0);
			return;
		}
		assert(buffer.size > 0);

		Alloc(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::Alloc(MemoryBuffer& buffer)
	{
		switch (ms)
		{
		case MemorySpace::Device:
			dm::detail::Alloc(buffer);
			break;
		case MemorySpace::Host:
			dm::detail::AllocHost(buffer);
			break;
		default:
			throw NotSupportedException();
		}
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename biRhs, MemorySpace msRhs, MathDomain mdRhs>
	void IBuffer<bi, ms, md>::ReadFrom(const IBuffer<biRhs, msRhs, mdRhs>& rhs)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);
		dm::detail::AutoCopy(buffer, static_cast<const bi*>(&rhs)->buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename T>
	void IBuffer<bi, ms, md>::ReadFrom(const std::vector<T>& rhs)
	{
		static_assert((std::is_same<T, double>::value && md == MathDomain::Double)
						||
					  (std::is_same<T, float>::value && md == MathDomain::Float)
						||
					  (std::is_same<T, int>::value && md == MathDomain::Int));
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		MemoryBuffer rhsBuf;
		ptr_t pointer = (ptr_t)(rhs.data());
		rhsBuf = MemoryBuffer(pointer, static_cast<unsigned>(rhs.size()), MemorySpace::Host, _Traits<T>::clType);

		dm::detail::AutoCopy(buffer, rhsBuf);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::Set(const stdType value) const
	{
		dm::detail::Initialize(buffer, value);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::LinSpace(const stdType x0, const stdType x1) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);
		dm::detail::LinSpace(buffer, x0, x1);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::RandomUniform(const unsigned seed) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);
		dm::detail::RandUniform(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::RandomGaussian(const unsigned seed) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);
		dm::detail::RandNormal(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> IBuffer<bi, ms, md>::Get() const
	{
		dm::detail::ThreadSynchronize();

		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		MemoryBuffer newBuf(buffer);
		newBuf.memorySpace = MemorySpace::Host;

		dm::detail::AllocHost(newBuf);
		dm::detail::AutoCopy(newBuf, buffer);

		std::vector<typename Traits<md>::stdType> ret(buffer.size);
		detail::Fill<md>(ret, newBuf);

		dm::detail::FreeHost(newBuf);

		return ret;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>::~IBuffer()
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dtor(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::dtor(MemoryBuffer buffer)
	{
		// if this is not the owner of the buffer, it must not free it
		if (!isOwner)
			return;
		assert(buffer.pointer != 0);

		switch (buffer.memorySpace)
		{
		case MemorySpace::Device:
			dm::detail::Free(buffer);
			break;
		case MemorySpace::Host:
			dm::detail::FreeHost(buffer);
			break;
		default:
			throw NotSupportedException();
		}
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename biRhs, MemorySpace msRhs, MathDomain mdRhs>
	bool IBuffer<bi, ms, md>::operator==(const IBuffer<biRhs, msRhs, mdRhs>& rhs) const
	{
		const auto& thisBuffer = Get();
		const auto& thatBuffer = rhs.Get();

		if (thisBuffer.size() != thatBuffer.size())
			return false;

		constexpr double tolerance = GetTolerance();
		for (size_t i = 0; i < thisBuffer.size(); ++i)
		{
			if (fabs(thisBuffer[i] - thatBuffer[i]) > tolerance)
				return false;
		}

		return true;
	}

	#pragma region Linear Algebra

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator +=(const IBuffer& rhs)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		dm::detail::AddEqual(buffer, static_cast<const bi*>(&rhs)->buffer, 1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator -=(const IBuffer& rhs)
	{
		assert(size() == rhs.size());
		assert(rhs.pointer != 0);

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		dm::detail::AddEqual(buffer, static_cast<bi>(rhs).buffer, -1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator %=(const IBuffer& rhs)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		MemoryBuffer tmp(0, buffer.size, ms, md);
		ctor(tmp);
		dm::detail::AutoCopy(tmp, buffer);

		dm::detail::ElementwiseProduct(buffer, tmp, static_cast<const bi*>(&rhs)->buffer, 1.0);

		dtor(tmp);

		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::AddEqual(const IBuffer& rhs, const double alpha)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		dm::detail::AddEqual(buffer, static_cast<const bi*>(&rhs)->buffer, alpha);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::Scale(const double alpha)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		dm::detail::Scale(buffer, alpha);
		return *this;
	}

#pragma endregion

	template<typename bi, MemorySpace ms, MathDomain md>
	void Print(const IBuffer<bi, ms, md>& buf, const std::string& label)
	{
		buf.Print(label);
	}

	template<typename T>
	static void Print(const std::vector<T>& vec, const std::string& label)
	{
		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t i = 0; i < vec.size(); i++)
			std::cout << "\tv[" << i << "] \t=\t " << vec[i] << std::endl;
		std::cout << "**********************" << std::endl;
	}

	template<typename T>
	static void Print(const std::vector<T>& mat, const unsigned nRows, const unsigned nCols, const std::string& label)
	{
		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t j = 0; j < nCols; j++)
		{
			std::cout << "\t";
			for (size_t i = 0; i < nRows; i++)
				std::cout << " m[" << i << "][" << j << "] = " << mat[i + nRows * j];
			std::cout << std::endl;
		}
		std::cout << "**********************" << std::endl;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Scale(IBuffer<bi, ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}
}
