#pragma once

namespace cl
{
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<ms, MathDomain::Int>&& nonZeroColumnIndices_, Vector<ms, MathDomain::Int>&& nNonZeroRows_)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false),  // CompressedSparseRowMatrix doesn't allocate its memory in its _buffer!
		_buffer(0, nonZeroColumnIndices_.size(), 0, 0, nRows, nCols, ms, md),
		values(nonZeroColumnIndices_.size()),
		nonZeroColumnIndices(std::move(nonZeroColumnIndices_)),
		nNonZeroRows(std::move(nNonZeroRows_))
	{
		SyncPointers();
	}
	
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<ms, MathDomain::Int>& nonZeroColumnIndices_, const Vector<ms, MathDomain::Int>& nNonZeroRows_)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false),  // CompressedSparseRowMatrix doesn't allocate its memory in its _buffer!
		_buffer(0, nonZeroColumnIndices_.size(), 0, 0, nRows, nCols, ms, md),
		values(nonZeroColumnIndices_.size()),
		nonZeroColumnIndices(nonZeroColumnIndices_),
		nNonZeroRows(nNonZeroRows_)
	{
		SyncPointers();
	}
	
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<ms, MathDomain::Int>&& nonZeroColumnIndices_, Vector<ms, MathDomain::Int>&& nNonZeroRows_, const typename Traits<md>::stdType value)
			: CompressedSparseRowMatrix(nRows, nCols, std::move(nonZeroColumnIndices_), std::move(nNonZeroRows_))
	{
		dm::detail::Initialize(values.GetBuffer(), static_cast<double>(value));
	}
	
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<ms, MathDomain::Int>& nonZeroColumnIndices_, const Vector<ms, MathDomain::Int>& nNonZeroRows_, const typename Traits<md>::stdType value)
		: CompressedSparseRowMatrix(nRows, nCols, nonZeroColumnIndices_, nNonZeroRows_)
	{
		dm::detail::Initialize(values.GetBuffer(), static_cast<double>(value));
	}

	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const ColumnWiseMatrix<ms, md>& denseMatrix)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false), _buffer(0, 0, 0, 0, denseMatrix.nRows(), denseMatrix.nCols(), ms, md)
	{
		ReadFrom(denseMatrix.Get(), denseMatrix.nRows(), denseMatrix.nCols());
	}
	
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false), _buffer(0, 0, 0, 0, static_cast<unsigned>(nRows), static_cast<unsigned>(nCols), ms, md)
	{
		ReadFrom(denseMatrix, nRows, nCols);
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::ReadFrom(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols)
	{
		std::vector<typename Traits<md>::stdType> nonZeroValues {};
		std::vector<int> _nonZeroColumnIndices {};
		std::vector<int> _nNonZeroRows {};
		int nNonZeros = 0;
		for (unsigned i = 0; i < nRows; i++)
		{
			for (unsigned j = 0; j < nCols; j++)
			{
				if (fabs(static_cast<double>(denseMatrix[i + j * nRows])) > 1e-7)
				{
					nNonZeros++;
					nonZeroValues.push_back(denseMatrix[i + j * nRows]);
					_nonZeroColumnIndices.push_back(static_cast<int>(j));
				}
			}
			
			_nNonZeroRows.push_back(nNonZeros);
		}
		
		_buffer.size = static_cast<unsigned>(nNonZeros);
		
		values._buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroValues.size()), ms, md);
		Alloc(values._buffer);
		values.ReadFrom(nonZeroValues);
		
		nonZeroColumnIndices._buffer = MemoryBuffer(0, static_cast<unsigned>(_nonZeroColumnIndices.size()), ms, md);
		Alloc(nonZeroColumnIndices._buffer);
		nonZeroColumnIndices.ReadFrom(_nonZeroColumnIndices);
		
		nNonZeroRows._buffer = MemoryBuffer(0, static_cast<unsigned>(_nNonZeroRows.size()), ms, md);
		Alloc(nNonZeroRows._buffer);
		nNonZeroRows.ReadFrom(_nNonZeroRows);
		
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const CompressedSparseRowMatrix& rhs)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false), // CompressedSparseRowMatrix doesn't allocate its memory in its _buffer!
		_buffer(0, 0, 0, 0, rhs.nRows(), rhs.nCols(), ms, md),
		values(rhs.values), nonZeroColumnIndices(rhs.nonZeroColumnIndices), nNonZeroRows(rhs.nNonZeroRows)
	{
		SyncPointers();
	}

template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(CompressedSparseRowMatrix&& rhs) noexcept
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false),
		  _buffer(rhs._buffer),
		  values(std::move(rhs.values)), nonZeroColumnIndices(std::move(rhs.nonZeroColumnIndices)), nNonZeroRows(std::move(rhs.nNonZeroRows))
	{
		rhs._isOwner = false;
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::SyncPointers()
	{
		_buffer.pointer = values._buffer.pointer;
		_buffer.nonZeroColumnIndices = nonZeroColumnIndices._buffer.pointer;
		_buffer.nNonZeroRows = nNonZeroRows._buffer.pointer;
	}

	template< MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> CompressedSparseRowMatrix<ms, md>::Get() const
	{
		auto _values = this->values.Get();
		auto _nonZeroColumnIndices = this->nonZeroColumnIndices.Get();
		auto _nNonZeroRows = this->nNonZeroRows.Get();

		std::vector<typename Traits<md>::stdType> ret(denseSize());
		size_t nz = 0;
		for (size_t i = 0; i < nRows(); i++)
		{
			const size_t nNonZeroInRow = static_cast<size_t>(_nNonZeroRows[i + 1] - _nNonZeroRows[i]);
			for (size_t j = 0; j < nNonZeroInRow; j++)
			{
				ret[i + static_cast<size_t>(_nonZeroColumnIndices[nz]) * nRows()] = _values[nz];
				nz++;
			}
		}

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::Print(const std::string& label) const
	{
		auto mat = Get();
		cl::Print(mat, nRows(), nCols(), label);
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::PrintNonZeros(const std::string& label) const
	{
		auto vec = this->values.Get();
		cl::Print(vec, label + " - values");
		
		auto nzci = this->nonZeroColumnIndices.Get();
		cl::Print(nzci, label + " - nonZeroColumnIndices");

		auto nnz = this->nNonZeroRows.Get();
		cl::Print(nnz, label + " - nnz");
	}

#pragma region Linear Algebra

	template< MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> CompressedSparseRowMatrix<ms, md>::operator *(const ColumnWiseMatrix<ms, md>& rhs) const
	{
		assert(nCols() == rhs.nRows());

		ColumnWiseMatrix<ms, md> ret(nRows(), rhs.nCols());
		dm::detail::SparseMultiply(ret._buffer, this->_buffer, rhs._buffer, MatrixOperation::None, 1.0);

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	Vector<ms, md> CompressedSparseRowMatrix<ms, md>::operator *(const Vector<ms, md>& rhs) const
	{
		assert(nRows() == rhs.size());

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseDot(ret._buffer, this->_buffer, rhs._buffer);

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> CompressedSparseRowMatrix<ms, md>::Multiply(const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha) const
	{
		ColumnWiseMatrix<ms, md> ret(nRows(), rhs.nCols());
		Multiply(ret, rhs, lhsOperation, alpha);

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha) const
	{
		assert(nCols() == rhs.nRows());
		dm::detail::SparseMultiply(out._buffer, this->_buffer, rhs._buffer, this->nRows(), rhs.nRows(), lhsOperation, alpha);
	}

	template< MemorySpace ms, MathDomain md>
	Vector<ms, md> CompressedSparseRowMatrix<ms, md>::Dot(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha) const
	{
		Vector<ms, md> ret(rhs.size());
		Dot(ret, rhs, lhsOperation, alpha);

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::Dot(Vector<ms, md>& out, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha) const
	{
		assert(nRows() == rhs.size());
		dm::detail::SparseDot(out._buffer, this->_buffer, rhs._buffer, lhsOperation, alpha);
	}

#pragma endregion 

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Multiply(const CompressedSparseRowMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		return lhs.Multiply(rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Multiply(ColumnWiseMatrix<ms, md>&out, const CompressedSparseRowMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		lhs.Multiply(out, rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Dot(const CompressedSparseRowMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		return lhs.Dot(rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Dot(Vector<ms, md>& out, const CompressedSparseRowMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		lhs.Dot(out, rhs, lhsOperation, alpha);
	}
}
