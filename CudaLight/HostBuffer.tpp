#pragma once

#include <iostream>
#include <iomanip>
#include <assert.h>

#include <Types.h>
#include <Npy++.h>
#include <HostRoutines/BufferInitializer.h>
#include <HostRoutines/MemoryManager.h>
#include <Exceptions.h>
#include <BlasWrappers.h>
#include <Extra.h>

namespace cl { namespace host
{
	template<typename bi, MemorySpace ms, MathDomain md>
	Buffer<bi, ms, md>::Buffer(const bool isOwner)
			: _isOwner(isOwner)
	{
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	Buffer<bi, ms, md>::Buffer(Buffer&& buf) noexcept
			: _isOwner(buf._isOwner)
	{
		buf._isOwner = false;  // otherwise the destructor will destroy the memory
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::ctor(MemoryBuffer& buffer)
	{
		// if is not the owner it has already been allocated!
		if (!_isOwner)
		{
			assert(buffer.pointer != 0);
			return;
		}
		assert(buffer.size > 0);

		Alloc(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::Alloc(MemoryBuffer& buffer)
	{
		routines::Alloc(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename biRhs, MemorySpace msRhs, MathDomain mdRhs>
	void Buffer<bi, ms, md>::ReadFrom(const Buffer<biRhs, msRhs, mdRhs>& rhs)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::Copy(buffer, static_cast<const bi*>(&rhs)->_buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename T>
	void Buffer<bi, ms, md>::ReadFrom(const std::vector<T>& rhs)
	{
		static_assert((std::is_same<T, double>::value && md == MathDomain::Double)
					  ||
					  (std::is_same<T, float>::value && md == MathDomain::Float)
					  ||
					  (std::is_same<T, int>::value && md == MathDomain::Int), "Invalid type");
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		MemoryBuffer rhsBuf;
		auto pointer = reinterpret_cast<ptr_t>(rhs.data());
		rhsBuf = MemoryBuffer(pointer, static_cast<unsigned>(rhs.size()), buffer.memorySpace, _Traits<T>::clType);

		routines::Copy(buffer, rhsBuf);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::Set(const stdType value)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::Initialize(buffer, static_cast<double>(value));
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::Reciprocal()
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::Reciprocal(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::LinSpace(const stdType x0, const stdType x1)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::LinSpace(buffer, static_cast<double>(x0), static_cast<double>(x1));
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::RandomUniform(const unsigned seed)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::RandUniform(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::RandomGaussian(const unsigned seed)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);
		routines::RandNormal(buffer, seed);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> Buffer<bi, ms, md>::Get() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		std::vector<typename Traits<md>::stdType> ret(buffer.size);
		detail::Fill<md>(ret, buffer);

		return ret;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Buffer<bi, ms, md>::dtor(MemoryBuffer& buffer)
	{
		// if this is not the owner of the buffer, it must not free it
		if (!_isOwner)
			return;
		assert(buffer.pointer != 0);
		routines::Free(buffer);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	template<typename biRhs, MemorySpace msRhs, MathDomain mdRhs>
	bool Buffer<bi, ms, md>::operator==(const Buffer<biRhs, msRhs, mdRhs>& rhs) const
	{
		const auto& thisBuffer = Get();
		const auto& thatBuffer = rhs.Get();

		if (thisBuffer.size() != thatBuffer.size())
			return false;

		constexpr double tolerance = GetTolerance();
		for (size_t i = 0; i < thisBuffer.size(); ++i)
		{
			if (std::fabs(thisBuffer[i] - thatBuffer[i]) > static_cast<stdType>(tolerance))
				return false;
		}

		return true;
	}

#pragma region Linear Algebra

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::operator +=(const IBuffer<ms, md>& rhs)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::AddEqual(buffer, static_cast<const bi*>(&rhs)->_buffer, 1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::operator -=(const IBuffer<ms, md>& rhs)
	{
		assert(size() == rhs.size());
		assert(static_cast<const bi*>(&rhs)->_buffer.pointer != 0);

		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::AddEqual(buffer, static_cast<const bi&>(rhs)._buffer, -1.0);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::operator %=(const IBuffer<ms, md>& rhs)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::ElementwiseProduct(buffer, buffer, static_cast<const bi*>(&rhs)->_buffer, 1.0);

		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::ElementWiseProduct(const IBuffer<ms, md>& rhs, const double alpha)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::ElementwiseProduct(buffer, buffer, static_cast<const bi*>(&rhs)->_buffer, alpha);

		return *this;
	}


	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::AddEqual(const IBuffer<ms, md>& rhs, const double alpha)
	{
		assert(size() == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);

		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::AddEqual(buffer, static_cast<const bi*>(&rhs)->_buffer, alpha);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	IBuffer<ms, md>& Buffer<bi, ms, md>::Scale(const double alpha)
	{
		MemoryBuffer& buffer = static_cast<bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		routines::Scale(buffer, alpha);
		return *this;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	int Buffer<bi, ms, md>::AbsoluteMinimumIndex() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		int ret = -1;
		routines::ArgAbsMin(ret, buffer);

		return ret;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	int Buffer<bi, ms, md>::AbsoluteMaximumIndex() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		int ret = -1;
		routines::ArgAbsMax(ret, buffer);

		return ret;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::AbsoluteMinimum() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		routines::AbsMin(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::AbsoluteMaximum() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		routines::AbsMax(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::Minimum() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		routines::Min(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::Maximum() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = 0.0;
		routines::Max(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::Sum() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = -1;
		routines::Sum(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	typename Traits<md>::stdType Buffer<bi, ms, md>::EuclideanNorm() const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(buffer.pointer != 0);

		double ret = -1;
		routines::EuclideanNorm(ret, buffer);

		return static_cast<typename Traits<md>::stdType>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	int Buffer<bi, ms, md>::CountEquals(const IBuffer<ms, md>& rhs) const
	{
		MemoryBuffer cache {};
		cache.memorySpace = ms;
		cache.mathDomain = md;
		cache.size = size();
		routines::Alloc(cache);

		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(rhs.size() == size());
		assert(buffer.pointer != 0);
		assert(static_cast<const bi*>(&rhs)->_buffer.pointer != 0);

		bool needToFreeCache = cache.pointer == 0;
		if (needToFreeCache)
		{
			cache.memorySpace = ms;
			cache.mathDomain = md;
			cache.size = size();
			routines::Alloc(cache);
		}

		// calculate the difference
		routines::Subtract(cache, buffer, static_cast<const bi*>(&rhs)->_buffer);

		// calculate how many non-zeros, overriding cache
		routines::IsNonZero(cache, cache);

		double ret = -1;
		routines::Sum(ret, cache);

		// we are counting the zero entries
		ret = size() - ret;

		if (needToFreeCache)
			routines::Free(cache);

		return static_cast<int>(ret);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	int Buffer<bi, ms, md>::CountEquals(const IBuffer<ms, md>& rhs, MemoryBuffer& cacheCount, MemoryBuffer& /*cacheSum*/, MemoryBuffer& /*oneElementCache*/) const
	{
		const MemoryBuffer& buffer = static_cast<const bi*>(this)->_buffer;
		assert(rhs.size() == size());
		assert(buffer.pointer != 0);
		assert(static_cast<const bi*>(&rhs)->_buffer.pointer != 0);

		// calculate the difference
		routines::Subtract(cacheCount, buffer, static_cast<const bi*>(&rhs)->_buffer);

		// calculate how many non-zeros, overriding cache
		//dm::detail::IsNonZero(cacheCount, cacheCount);

		double ret = -1;
		//dm::detail::SumWithProvidedCache(ret, cacheCount, cacheSum, oneElementCache);

		// we are counting the zero entries
		ret = size() - ret;

		return static_cast<int>(ret);
	}

#pragma endregion

	template<typename bi, MemorySpace ms, MathDomain md>
	void Scale(Buffer<bi, ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const Buffer<bi, ms, md>& buffer)
	{
		buffer.ToOutputStream(os);
		return os;
	}

	template<typename bi, MemorySpace ms, MathDomain md>
	void Print(const Buffer<bi, ms, md>& buf, const std::string& label)
	{
		buf.Print(label);
	}

	template<typename T>
	void Print(const std::vector<T>& v, const std::string& label)
	{
		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t i = 0; i < v.size(); i++)
			std::cout << "\tv[" << i << "] \t=\t " << v[i] << std::endl;
		std::cout << "**********************" << std::endl;
	}

	template<typename T>
	void Print(const std::vector<T>& m, const unsigned nRows, const unsigned nCols, const std::string& label)
	{
		std::cout << "********* " << label << " ***********" << std::endl;

		for (size_t i = 0; i < nRows; i++)
		{
			std::cout << "\t";
			for (size_t j = 0; j < nCols; j++)
				std::cout << " m[" << i << "][" << j << "] = " << m[i + nRows * j];
			std::cout << std::endl;
		}
		std::cout << "**********************" << std::endl;
	}

#pragma region Serialization

	template<typename T>
	std::ostream& VectorToOutputStream(const std::vector<T>& v, std::ostream& os)
	{
		for (size_t i = 0; i < v.size(); i++)
			os << std::setprecision(16) << v[i] << std::endl;

		return os;
	}

	template<typename T>
	std::istream& VectorFromInputStream(std::vector<T>& v, std::istream& is)
	{
		T value;
		while (is >> value)
			v.push_back(value);

		return is;
	}

	template<typename T>
	std::ostream& MatrixToOutputStream(const std::vector<T>& m, const unsigned nRows, const unsigned nCols, std::ostream& os)
	{
		for (size_t i = 0; i < nRows; i++)
		{
			for (size_t j = 0; j < nCols; j++)
				os << std::setprecision(16) << m[i + nRows * j] << " ";
			os << std::endl;
		}

		return os;
	}

	template<typename T>
	std::istream& MatrixFromInputStream(std::vector<T>& m, unsigned& nRows, unsigned& nCols, std::istream& is)
	{
		std::string line;

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
		nCols = static_cast<unsigned>(matTranspose.size()) / nRows;

		// transpose matrix
		m.resize(matTranspose.size());
		for (size_t i = 0; i < nRows; i++)
		{
			for (size_t j = 0; j < nCols; j++)
				m[i + nRows * j] = matTranspose[j + nCols * i];
		}

		return is;
	}

	template<typename T>
	void VectorToBinaryFile(const std::vector<T>& v, const std::string& fileName,  const bool compressed, const std::string mode)
	{
		if (!compressed)
			npypp::Save(fileName, v, { v.size() }, mode);
		else
			npypp::SaveCompressed(fileName, v, { v.size() }, mode);
	}

	template<typename T>
	void VectorFromBinaryFile(std::vector<T>& v, const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		if (!compressed)
			v = npypp::Load<T>(fileName, useMemoryMapping);
		else
			v = npypp::LoadCompressed<T>(fileName).begin()->second;
	}

	template<typename T>
	std::vector<T> VectorFromBinaryFile(const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		std::vector<T> v;
		VectorFromBinaryFile(v, fileName, compressed, useMemoryMapping);

		return v;
	}

	template<typename T>
	void MatrixToBinaryFile(const std::vector<T>& m, unsigned nRows, unsigned nCols, const std::string& fileName, const bool tranpose, const bool compressed, const std::string mode)
	{
		std::vector<T> matTranspose {};
		if (tranpose)
		{
			matTranspose.resize(m.size());
			for (size_t i = 0; i < nRows; ++i)
			{
				for (size_t j = 0; j < nCols; ++j)
					matTranspose[i + j * nRows] = m[j + i * nCols];
			}
		}
		else
		{
			// cannot use std::swap, as this code might be compiled with nvcc
			auto tmp = nCols;
			nCols = nRows;
			nRows = tmp;
		}
		matTranspose = m;

		if (!compressed)
			npypp::Save(fileName, matTranspose, { static_cast<size_t>(nCols), static_cast<size_t>(nRows) }, mode);
		else
			npypp::SaveCompressed(fileName, matTranspose, { static_cast<size_t>(nCols), static_cast<size_t>(nRows) }, mode);
	}

	template<typename T>
	void MatrixFromBinaryFile(std::vector<T>& m, unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		const auto& fullExtract = !compressed ? npypp::LoadFull<T>(fileName, useMemoryMapping) : npypp::LoadCompressedFull<T>(fileName).begin()->second;

		assert(fullExtract.shape.size() == 2);
		nRows = static_cast<unsigned>(fullExtract.shape[0]);
		nCols = static_cast<unsigned>(fullExtract.shape[1]);

		m.resize(fullExtract.data.size());
		for (size_t i = 0; i < nRows; ++i)
		{
			for (size_t j = 0; j < nCols; ++j)
				m[i + j * nRows] = fullExtract.data[j + i * nCols];
		}
	}

	template<typename T>
	std::vector<T> MatrixFromBinaryFile(unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		std::vector<T> m;
		MatrixFromBinaryFile(m, nRows, nCols, fileName, compressed, useMemoryMapping);

		return m;
	}

#pragma endregion
}}
