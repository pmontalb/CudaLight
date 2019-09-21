#pragma once

namespace cl
{
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<ms, MathDomain::Int>& nonZeroColumnIndices, const Vector<ms, MathDomain::Int>& nNonZeroRows)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false),  // CompressedSparseRowMatrix doesn't allocate its memory in its _buffer!
		_buffer(0, nonZeroColumnIndices.size(), 0, 0, nRows, nCols, ms, md),
		values(nonZeroColumnIndices.size()), nonZeroColumnIndices(nonZeroColumnIndices), nNonZeroRows(nNonZeroRows)
	{
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<ms, MathDomain::Int>& nonZeroColumnIndices, const Vector<ms, MathDomain::Int>& nNonZeroRows, const typename Traits<md>::stdType value)
		: CompressedSparseRowMatrix(nRows, nCols, nonZeroColumnIndices, nNonZeroRows)
	{
		dm::detail::Initialize(values.GetBuffer(), value);
	}

	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const ColumnWiseMatrix<ms, md>& denseMatrix)
		: IBuffer<CompressedSparseRowMatrix<ms, md>, ms, md>(false), _buffer(0, 0, 0, 0, denseMatrix.nRows(), denseMatrix.nCols(), ms, md)
	{
		const auto hostDenseMatrix = denseMatrix.Get();

		std::vector<typename Traits<md>::stdType> nonZeroValues;
		std::vector<int> nonZeroColumnIndices;
		std::vector<int> nNonZeroRows;
		int nNonZeros = 0;
		for (unsigned i = 0; i < denseMatrix.nRows(); i++)
		{
			for (unsigned j = 0; j < denseMatrix.nCols(); j++)
			{
				if (fabs(hostDenseMatrix[i + j * nRows()]) > 1e-7)
				{
					nNonZeros++;
					nonZeroValues.push_back(hostDenseMatrix[i + j * nRows()]);
					nonZeroColumnIndices.push_back(j);
				}
			}

			nNonZeroRows.push_back(nNonZeros);
		}

		_buffer.size = nNonZeros;

		values._buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroValues.size()), ms, md);
		Alloc(values._buffer);
		values.ReadFrom(nonZeroValues);

		this->nonZeroColumnIndices._buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroColumnIndices.size()), ms, md);
		Alloc(this->nonZeroColumnIndices._buffer);
		this->nonZeroColumnIndices.ReadFrom(nonZeroColumnIndices);

		this->nNonZeroRows._buffer = MemoryBuffer(0, static_cast<unsigned>(nNonZeroRows.size()), ms, md);
		Alloc(this->nNonZeroRows._buffer);
		this->nNonZeroRows.ReadFrom(nNonZeroRows);

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
	void CompressedSparseRowMatrix<ms, md>::SyncPointers()
	{
		_buffer.pointer = values._buffer.pointer;
		_buffer.nonZeroColumnIndices = nonZeroColumnIndices._buffer.pointer;
		_buffer.nNonZeroRows = nNonZeroRows._buffer.pointer;
	}

	template< MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> CompressedSparseRowMatrix<ms, md>::Get() const
	{
		auto values = this->values.Get();
		auto nonZeroColumnIndices = this->nonZeroColumnIndices.Get();
		auto nNonZeroRows = this->nNonZeroRows.Get();

		std::vector<typename Traits<md>::stdType> ret(denseSize());
		int nz = 0;
		for (unsigned i = 0; i < nRows(); i++)
		{
			const int nNonZeroInRow = nNonZeroRows[i + 1] - nNonZeroRows[i];
			for (int j = 0; j < nNonZeroInRow; j++)
			{
				ret[i + nonZeroColumnIndices[nz] * nRows()] = values[nz];
				nz++;
			}
		}

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::Print(const std::string& label) const
	{
		auto mat = Get();
		cl::Print(mat, label);
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
		dm::detail::SparseDot(out._buffer, this->_buffer, rhs._buffer, lhsOperation, 1.0);
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