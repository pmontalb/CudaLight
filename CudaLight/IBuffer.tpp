#pragma once

#include <iostream>
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
			if (!buffer.pointer)
				throw BufferNotInitialisedException("Pointer must be allocated first!");
			return;
		}

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
		if (!buffer.pointer)
			throw BufferNotInitialisedException("Buffer needs to be initialised first!");

		dm::detail::AutoCopy(buffer, rhs.buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::LinSpace(const double x0, const double x1) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		dm::detail::LinSpace(buffer, x0, x1);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::RandomUniform(const unsigned seed) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		dm::detail::RandUniform(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void IBuffer<bi, ms, md>::RandomGaussian(const unsigned seed) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		dm::detail::RandNormal(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>::~IBuffer()
	{
		// if this is not the owner of the buffer, it must not free it
		if (!isOwner)
			return;

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
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

	#pragma region Linear Algebra

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator +=(const IBuffer& rhs)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dm::detail::AddEqual(buffer, static_cast<bi>(rhs).buffer, 1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator -=(const IBuffer& rhs)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dm::detail::AddEqual(buffer, static_cast<bi>(rhs).buffer, -1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::operator %=(const IBuffer& rhs)
	{
		const IBuffer tmp(*this);
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dm::detail::ElementwiseProduct(buffer, static_cast<bi>(tmp).buffer, static_cast<bi>(rhs).buffer, 1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::AddEqual(const IBuffer& rhs, const double alpha)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dm::detail::AddEqual(buffer, static_cast<bi>(rhs).buffer, alpha);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<bi, ms, md>& IBuffer<bi, ms, md>::Scale(const double alpha)
	{
		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		dm::detail::Scale(buffer, alpha);
		return *this;
	}

#pragma endregion

	template<typename bi, MemorySpace ms, MathDomain md>
	void Print(const IBuffer<bi, ms, md>& buf, const std::string& label)
	{
		buf.Print(label);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Scale(IBuffer<bi, ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}
}
