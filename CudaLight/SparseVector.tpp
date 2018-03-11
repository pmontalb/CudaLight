#pragma once

namespace cl
{
	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices)
		: IBuffer(false),  // SparseVector doesn't allocate its memory in its buffer!
		denseSize(size),
		buffer(0, nonZeroIndices.size(), 0, ms, md),
		values(nonZeroIndices.size()), nonZeroIndices(nonZeroIndices)
	{
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices, const typename Traits<md>::stdType value)
		: SparseVector(size, nonZeroIndices)
	{
		dm::detail::Initialize(values.GetBuffer(), value);
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const unsigned nNonZeros)
		: IBuffer(false), // SparseVector doesn't allocate its memory in its buffer!
		  denseSize(size), values(nNonZeros), nonZeroIndices(nNonZeros),
		{
			assert(size != 0);
			assert(nNonZeros != 0);
			assert(nNonZeros < size);

			SyncPointers();
		}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const unsigned nNonZeros, const typename Traits<md>::stdType value)
		: SparseVector(size, nNonZeros)
	{
		dm::detail::Initialize(static_cast<MemoryBuffer>(buffer), value);
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const Vector<ms, md>& denseVector)
		: IBuffer(false), denseSize(denseVector.size())
	{
		const auto hostDenseVector = denseVector.Get();

		std::vector<int> nonZeroIndices;
		std::vector<typename Traits<md>::stdType> nonZeroValues;
		for (int i = 0; i < hostDenseVector.size(); ++i)
		{
			if (fabs(hostDenseVector[i]) > 1e-7)
			{
				nonZeroIndices.push_back(i);
				nonZeroValues.push_back(hostDenseVector[i]);
			}
		}

		values.buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroValues.size()), ms, md);
		Alloc(values.buffer);
		values.ReadFrom(nonZeroValues);

		this->nonZeroIndices.buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroIndices.size()), ms, md);
		Alloc(this->nonZeroIndices.buffer);
		this->nonZeroIndices.ReadFrom(nonZeroIndices);

		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const SparseVector& rhs)
		: IBuffer(false), // SparseVector doesn't allocate its memory in its buffer!
		values(rhs.values), nonZeroIndices(rhs.nonZeroIndices), denseSize(rhs.denseSize)
	{
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	void SparseVector<ms, md>::SyncPointers()
	{
		buffer.pointer = values.buffer.pointer;
		buffer.indices = nonZeroIndices.buffer.pointer;
	}

	template< MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> SparseVector<ms, md>::Get() const
	{
		auto values = this->values.Get();
		auto indices = this->nonZeroIndices.Get();

		std::vector<typename Traits<md>::stdType> ret(denseSize);
		for (int i = 0; i < indices.size(); i++)
			ret[indices[i]] = values[i];

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void SparseVector<ms, md>::Print(const std::string& label) const
	{
		auto vec = Get();
		cl::Print(vec, label);
	}

	#pragma region Dense-Sparse Linear Algebra

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> SparseVector<ms, md >::operator +(const Vector<ms, md>& rhs) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseAdd(ret.buffer, buffer, rhs.buffer, 1.0);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> SparseVector<ms, md>::operator -(const Vector<ms, md>& rhs) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseSubtract(ret.buffer, buffer, rhs.buffer);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>SparseVector<ms, md>::Add(const Vector<ms, md>& rhs, const double alpha) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseAdd(ret.buffer, buffer, rhs.buffer, alpha);
		return ret;
	}

	#pragma endregion

	#pragma region Linear Algebra

	template<MemorySpace ms, MathDomain md>
	SparseVector<ms, md> SparseVector<ms, md>::operator +(const SparseVector& rhs) const
	{
		SparseVector<ms, md> ret(*this);
		ret += rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	SparseVector<ms, md> SparseVector<ms, md>::operator -(const SparseVector& rhs) const
	{
		SparseVector<ms, md> ret(*this);
		ret -= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	SparseVector<ms, md> SparseVector<ms, md>::operator %(const SparseVector<ms, md>& rhs) const
	{
		SparseVector<ms, md> ret(*this);
		ret %= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	SparseVector<ms, md> SparseVector<ms, md>::Add(const SparseVector<ms, md>& rhs, const double alpha) const
	{
		Vector<ms, md> ret(*this);
		ret.AddEqual(rhs, alpha);

		return ret;
	}

	#pragma endregion
}