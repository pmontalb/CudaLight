#pragma once

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <limits>

#include <Types.h>
#include <Npy++.h>

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
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);
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
		assert(static_cast<const bi*>(&rhs)->buffer.pointer != 0);

		const MemoryBuffer& buffer = static_cast<bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		dm::detail::AddEqual(buffer, static_cast<const bi&>(rhs).buffer, -1.0);
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

	template<typename bi, MemorySpace ms, MathDomain md>
	int IBuffer<bi, ms, md>::AbsoluteMinimumIndex() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		int ret = -1;
		dm::detail::ArgAbsMin(ret, buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	int IBuffer<bi, ms, md>::AbsoluteMaximumIndex() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		int ret = -1;
		dm::detail::ArgAbsMax(ret, buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType IBuffer<bi, ms, md>::MinimumInAbsoluteValue() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		dm::detail::AbsMin(ret, buffer);

		return ret;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType IBuffer<bi, ms, md>::MaximumInAbsoluteValue() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		dm::detail::AbsMax(ret, buffer);

		return ret;
	}

#pragma endregion

	template<typename bi, MemorySpace ms, MathDomain md>
	void Scale(IBuffer<bi, ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const IBuffer<bi, ms, md>& buffer)
	{
		buffer.ToOutputStream(os);
		return os;
	}

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
		
		for (size_t i = 0; i < nRows; i++)
		{
			std::cout << "\t";
			for (size_t j = 0; j < nCols; j++)
				std::cout << " m[" << i << "][" << j << "] = " << mat[i + nRows * j];
			std::cout << std::endl;
		}
		std::cout << "**********************" << std::endl;
	}

    #pragma region Serialization

	template<typename T>
	static std::ostream& VectorToOutputStream(const std::vector<T>& vec, std::ostream& os)
	{
		for (size_t i = 0; i < vec.size(); i++)
			os << std::setprecision(16) << vec[i] << std::endl;

		return os;
	}

	template<typename T>
	static std::istream& VectorFromInputStream(std::vector<T>& vec, std::istream& is)
	{
		T value;
		while (is >> value)
			vec.push_back(value);

		return is;
	}

	template<typename T>
	static std::ostream& MatrixToOutputStream(const std::vector<T>& mat, const unsigned nRows, const unsigned nCols, std::ostream& os)
	{
		for (size_t i = 0; i < nRows; i++)
		{
			for (size_t j = 0; j < nCols; j++)
				os << std::setprecision(16) << mat[i + nRows * j] << " ";
			os << std::endl;
		}

		return os;
	}

	template<typename T>
	static std::istream& MatrixFromInputStream(std::vector<T>& mat, unsigned& nRows, unsigned& nCols, std::istream& is)
	{
		std::string line;	
		unsigned i = 0;

		// read transposed matrix
		std::vector<T> matTranspose;
		while (std::getline(is, line))
		{
			std::stringstream ss(line);

			T value;
			while (ss >> value)
				matTranspose.push_back(value);
			++nRows;
		}
		nCols = matTranspose.size() / nRows;

		// transpose matrix
		mat.resize(matTranspose.size());
		for (size_t i = 0; i < nRows; i++)
		{
			for (size_t j = 0; j < nCols; j++)
				mat[i + nRows * j] = matTranspose[j + nCols * i];
		}

		return is;
	}

	template<typename T>
	static void VectorToBinaryFile(const std::vector<T>& vec, const std::string& fileName, const std::string mode)
	{
		npypp::Save(fileName, vec, { vec.size() }, mode);
	}

	template<typename T>
	static void VectorFromBinaryFile(std::vector<T>& vec, const std::string& fileName, const bool useMemoryMapping)
	{
		vec = npypp::Load<T>(fileName, useMemoryMapping);
	}

	template<typename T>
	static void MatrixToBinaryFile(const std::vector<T>& mat, const unsigned nRows, const unsigned nCols, const std::string& fileName, const std::string mode)
	{
		npypp::Save(fileName, mat, { static_cast<size_t>(nRows), static_cast<size_t>(nCols) }, mode);
	}

	template<typename T>
	static void MatrixFromBinaryFile(std::vector<T>& mat, unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool useMemoryMapping)
	{
		const auto& fullExtract = npypp::LoadFull<T>(fileName, useMemoryMapping);
		mat = fullExtract.data;

		assert(fullExtract.shape.size() == 2);
		nRows = fullExtract.shape[0];
		nCols = fullExtract.shape[1];
	}

    #pragma endregion
}
