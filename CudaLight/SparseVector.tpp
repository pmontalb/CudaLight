#pragma once

namespace cl
{
	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices_)
		: Buffer<SparseVector < ms, md>, ms, md>(false),  // SparseVector doesn't allocate its memory in its _buffer!
		denseSize(size),
		_buffer(0, nonZeroIndices_.size(), 0, ms, md),
		values(nonZeroIndices_.size()), nonZeroIndices(nonZeroIndices_)
	{
		SyncPointers();
	}
	
	template<MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(SparseVector&& rhs) noexcept
			: Buffer<SparseVector < ms, md>, ms, md>(false), _buffer(rhs._buffer)
	{
		rhs._isOwner = false;
		SyncPointers();
	}
	
	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices_, const typename Traits<md>::stdType value)
		: SparseVector(size, nonZeroIndices_)
	{
		dm::detail::Initialize(values.GetBuffer(), static_cast<double>(value));
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const unsigned size, const unsigned nNonZeros)
		: Buffer<SparseVector < ms, md>, ms, md>(false), // SparseVector doesn't allocate its memory in its _buffer!
		  denseSize(size), values(nNonZeros), nonZeroIndices(nNonZeros)
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
		dm::detail::Initialize(static_cast<MemoryBuffer>(_buffer), value);
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const Vector<ms, md>& denseVector)
		: Buffer<SparseVector < ms, md>, ms, md>(false), denseSize(denseVector.size())
	{
		const auto hostDenseVector = denseVector.Get();

		std::vector<int> _nonZeroIndices;
		std::vector<typename Traits<md>::stdType> nonZeroValues;
		for (size_t i = 0; i < hostDenseVector.size(); ++i)
		{
			if (std::fabs(hostDenseVector[i]) > static_cast<typename Traits<md>::stdType>(1e-7))
			{
                _nonZeroIndices.push_back(static_cast<int>(i));
				nonZeroValues.push_back(hostDenseVector[i]);
			}
		}

		values._buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroValues.size()), ms, md);
		Alloc(values._buffer);
		values.ReadFrom(nonZeroValues);

		nonZeroIndices._buffer = MemoryBuffer(0, static_cast<unsigned>(_nonZeroIndices.size()), ms, md);
		Alloc(this->nonZeroIndices._buffer);
		nonZeroIndices.ReadFrom(_nonZeroIndices);

		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	SparseVector<ms, md>::SparseVector(const SparseVector& rhs)
		: Buffer<SparseVector < ms, md>, ms, md>(false), // SparseVector doesn't allocate its memory in its _buffer!
		denseSize(rhs.denseSize), values(rhs.values), nonZeroIndices(rhs.nonZeroIndices)
{
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	void SparseVector<ms, md>::SyncPointers()
	{
		_buffer.pointer = values._buffer.pointer;
		_buffer.indices = nonZeroIndices._buffer.pointer;
	}

	template< MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> SparseVector<ms, md>::Get() const
	{
		auto _values = this->values.Get();
		auto indices = this->nonZeroIndices.Get();

		std::vector<typename Traits<md>::stdType> ret(denseSize);
		for (size_t i = 0; i < indices.size(); i++)
			ret[static_cast<size_t>(indices[i])] = _values[i];

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void SparseVector<ms, md>::Print(const std::string& label) const
	{
		auto v = Get();
		cl::Print(v, label);
	}

	#pragma region Dense-Sparse Linear Algebra

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> SparseVector<ms, md >::operator +(const Vector<ms, md>& rhs) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(_buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseAdd(ret._buffer, _buffer, rhs._buffer, 1.0);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> SparseVector<ms, md>::operator -(const Vector<ms, md>& rhs) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(_buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseSubtract(ret._buffer, _buffer, rhs._buffer);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md>SparseVector<ms, md>::Add(const Vector<ms, md>& rhs, const double alpha) const
	{
		assert(denseSize == rhs.size());
		assert(rhs.GetBuffer().pointer != 0);
		assert(_buffer.pointer != 0);

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseAdd(ret._buffer, _buffer, rhs._buffer, alpha);
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
