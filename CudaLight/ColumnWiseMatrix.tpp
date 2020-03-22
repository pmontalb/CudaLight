#pragma once

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols)
		: Buffer<ColumnWiseMatrix < ms, md>, ms, md>(true), _buffer(0, nRows, nCols, ms, md)
	{
		this->ctor(_buffer);
	
		SetUp(nCols);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(ColumnWiseMatrix&& rhs) noexcept
			: Buffer<ColumnWiseMatrix<ms, md>, ms, md>(std::move(rhs)), columns(std::move(rhs.columns)), _buffer(rhs._buffer)
	{
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::SetUp(const size_t nCols)
	{
		columns.resize(nCols);
		for (size_t i = 0; i < nCols; i++)
		{
			const size_t colShift = i * nRows() * _buffer.ElementarySize();
			MemoryBuffer colBuffer(_buffer.pointer + colShift, _buffer.nRows, ms, md);
			columns[i] = Vector<ms, md>::make_shared(colBuffer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols, const typename Traits<md>::stdType value)
		: ColumnWiseMatrix(nRows, nCols)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Initialize(_buffer, static_cast<double>(value));
		else
			routines::Initialize(_buffer, static_cast<double>(value));
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows)
		: ColumnWiseMatrix(nRows, nRows)
	{
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const ColumnWiseMatrix& rhs)
		: ColumnWiseMatrix(rhs.nRows(), rhs.nCols())
	{
		ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const ColumnWiseMatrix& rhs, const size_t colStart, const size_t colEnd)
		: Buffer<ColumnWiseMatrix<ms, md>, ms, md>(false),
		  _buffer(rhs.GetTile())
	{
		assert(colStart < colEnd);
		assert(colEnd <= nCols());

		const size_t nCols = colEnd - colStart;
		_buffer.nCols = static_cast<unsigned>(nCols);
		_buffer.size = static_cast<unsigned>(_buffer.nRows * nCols);
		
		const size_t colStartShift = colStart * nRows() * _buffer.ElementarySize();
		_buffer.pointer += colStartShift;

		SetUp(nCols);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const Vector<ms, md>& rhs, const size_t startOffset, const size_t nRows, const size_t nCols)
		: Buffer<ColumnWiseMatrix<ms, md>, ms, md>(false),
		  _buffer(0, static_cast<unsigned>(nRows), static_cast<unsigned>(nCols), static_cast<unsigned>(nRows), ms, md)
	{
		assert(startOffset + nRows * nCols <= rhs.size());
		_buffer.pointer = rhs.GetBuffer().pointer + startOffset * rhs.GetBuffer().ElementarySize();
		SetUp(nCols);
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const std::string& fileName, bool useMemoryMapping)
	{
		std::vector<typename Traits<md>::stdType> m {};
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromBinaryFile(m, nRows, nCols, fileName, useMemoryMapping);
		
		ReadFrom(m, nRows, nCols);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const std::vector<typename Traits<md>::stdType>& rhs, const unsigned nRows, const unsigned nCols)
		: ColumnWiseMatrix(nRows, nCols)
	{
		assert(rhs.size() == nRows * nCols);
		ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const Vector<ms, md>& rhs)
		: ColumnWiseMatrix(rhs.size(), 1)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::AutoCopy(columns[0]->GetBuffer(), rhs.GetBuffer());
		else
			routines::Copy(columns[0]->GetBuffer(), rhs.GetBuffer());
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const MemoryTile& buffer)
		: Buffer<ColumnWiseMatrix<ms, md>, ms, md>(false), _buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::MakeIdentity()
	{
		assert(nRows() == nCols());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Eye(this->_buffer);
		else
			routines::Eye(this->_buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::Flatten() const
	{
		Vector<ms, md> ret(this->size());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::AutoCopy(ret.GetBuffer(), _buffer);
		else
			routines::Copy(ret.GetBuffer(), _buffer);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ReadFrom(const Vector<ms, md>& rhs)
	{
		assert(rhs.size() > 0);
		assert(rhs.GetBuffer().pointer != 0);
		assert(_buffer.pointer != 0);

		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::AutoCopy(columns[0]->buffer, rhs.GetBuffer());
		else
			routines::Copy(columns[0]->buffer, rhs.GetBuffer());
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> ColumnWiseMatrix<ms, md>::Get(const unsigned column) const
	{
		assert(column < nCols());
		return columns[column]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Set(const Vector<ms, md>& columnVector, const unsigned column)
	{
		assert(column < nCols());
		columns[column]->ReadFrom(columnVector);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::RandomShuffleColumns(const unsigned seed)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::RandShuffleColumns(this->_buffer, seed);
		else
			routines::RandShuffleColumns(this->_buffer, seed);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Print(const std::string& label) const
	{
		auto m = Get();
		cl::Print(m, nRows(), nCols(), label);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& ColumnWiseMatrix<ms, md>::ToOutputStream(std::ostream& os) const
	{
		cl::MatrixToOutputStream(Get(), nRows(), nCols(), os);
		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ToBinaryFile(const std::string& fileName, const bool compressed, const std::string mode) const
	{
		cl::MatrixToBinaryFile(Get(), nRows(), nCols(), fileName, false, compressed, mode);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const ColumnWiseMatrix<ms, md>& buffer)
	{
		buffer.ToOutputStream(os);
		return os;
	}

	#pragma region Linear Algebra

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ScaleColumns(const Vector<ms, md>& alpha)
	{
		assert(_buffer.pointer != 0);
		assert(nCols() == alpha.size());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::ScaleColumns(_buffer, alpha.GetBuffer());
		else
			routines::ScaleColumns(_buffer, alpha.GetBuffer());
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator +(const ColumnWiseMatrix& rhs) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());

		ColumnWiseMatrix ret(*this);
		ret += rhs;

		return ret;
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator -(const ColumnWiseMatrix& rhs) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());

		ColumnWiseMatrix ret(*this);
		ret -= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator %(const ColumnWiseMatrix& rhs) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());

		ColumnWiseMatrix ret(*this);
		ret %= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Add(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());

		ColumnWiseMatrix ret(*this);
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::AddEqualMatrix(ret._buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);
		else
			routines::AddEqualMatrix(ret._buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *(const ColumnWiseMatrix& rhs) const
	{
		assert(nCols() == rhs.nRows());

		ColumnWiseMatrix ret(nRows(), rhs.nCols());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Multiply(ret._buffer, this->_buffer, rhs._buffer, MatrixOperation::None, MatrixOperation::None, 1.0, 0.0);
		else
			routines::Multiply(ret._buffer, this->_buffer, rhs._buffer, MatrixOperation::None, MatrixOperation::None, 1.0, 0.0);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *=(const ColumnWiseMatrix& rhs)
	{
		assert(nCols() == rhs.nRows());

		ColumnWiseMatrix ret(nRows(), rhs.nCols());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
		{
			dm::detail::Multiply(ret.buffer, this->buffer, rhs.buffer, MatrixOperation::None, MatrixOperation::None, this->nRows(), rhs.nRows());
			dm::detail::AutoCopy(_buffer, ret.buffer);
		}
		else
		{
			routines::Multiply(ret.buffer, this->buffer, rhs.buffer, MatrixOperation::None, MatrixOperation::None, this->nRows(), rhs.nRows());
			routines::Copy(_buffer, ret.buffer);
		}
		return *this;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::operator *(const Vector<ms, md>& rhs) const
	{
		assert(nCols() == rhs.size());

		Vector<ms, md> ret(nRows());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Dot(ret.GetBuffer(), this->_buffer, rhs.GetBuffer());
		else
			routines::Dot(ret.GetBuffer(), this->_buffer, rhs.GetBuffer());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>& ColumnWiseMatrix<ms, md>::AddEqualMatrix(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta)
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());
		
		assert(_buffer.pointer != 0);

		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::AddEqualMatrix(_buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);
		else
			routines::AddEqualMatrix(_buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);
		return *this;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>& ColumnWiseMatrix<ms, md>::AddEqualBroadcast(const Vector<ms, md>& rhs, const bool rowWise, const double alpha)
	{
		const size_t onesSize = !rowWise ? nCols() : nRows();
		Vector<ms, md> ones(static_cast<unsigned>(onesSize), 1.0);
		AddEqualBroadcast(rhs, ones, rowWise, alpha);
		
		return *this;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>& ColumnWiseMatrix<ms, md>::AddEqualBroadcast(const Vector<ms, md>& rhs, const Vector<ms, md>& ones, const bool rowWise, const double alpha)
	{
		assert((rowWise && rhs.size() == nCols()) || (!rowWise && rhs.size() == nRows()));
		assert((rowWise && ones.size() == nRows()) || (!rowWise && ones.size() == nCols()));
		
		if (rowWise)
		{
			if (ms == MemorySpace::Host || ms == MemorySpace::Device)
				dm::detail::KroneckerProduct(_buffer, ones.GetBuffer(), rhs.GetBuffer(), alpha);
			else
				routines::KroneckerProduct(_buffer, ones.GetBuffer(), rhs.GetBuffer(), alpha);
		}
		else
		{
			if (ms == MemorySpace::Host || ms == MemorySpace::Device)
				dm::detail::KroneckerProduct(_buffer, rhs.GetBuffer(), ones.GetBuffer(), alpha);
			else
				routines::KroneckerProduct(_buffer, rhs.GetBuffer(), ones.GetBuffer(), alpha);
		}
		
		return *this;
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Multiply(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		unsigned nRows, nCols;
		if (lhsOperation == MatrixOperation::None)
			nRows = this->nRows();
		else
			nRows = this->nCols();
		if (rhsOperation == MatrixOperation::None)
			nCols = rhs.nCols();
		else
			nCols = rhs.nRows();
		ColumnWiseMatrix ret(nRows, nCols);
		Multiply(ret, rhs, lhsOperation, rhsOperation, alpha, beta);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Multiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		size_t nRowsEnd, nColsEnd, nColsRhsEnd;
		if (lhsOperation == MatrixOperation::None)
		{
			nRowsEnd = this->nRows();
			nColsEnd = this->nCols();
		}
		else
		{
			nRowsEnd = this->nCols();
			nColsEnd = this->nRows();
		}
		if (rhsOperation == MatrixOperation::None)
			nColsRhsEnd = rhs.nCols();
		else
			nColsRhsEnd = rhs.nRows();
		this->SubMultiply(out, rhs, 0, 0, nRowsEnd, nColsEnd, 0, nColsRhsEnd, lhsOperation, rhsOperation, alpha, beta);
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::SubMultiply(const ColumnWiseMatrix& rhs, const size_t rowStart, const size_t colStart, const size_t nRows, const size_t nCols, const size_t colRhsStart, const size_t nColsRhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		ColumnWiseMatrix ret(this->nRows(), rhs.nCols(), 0.0);
		SubMultiply(ret, rhs, rowStart, colStart, nRows, nCols, colRhsStart, nColsRhs, lhsOperation, rhsOperation, alpha, beta);
		
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::SubMultiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const size_t rowStart, const size_t colStart, const size_t nRows, const size_t nCols, const size_t colRhsStart, const size_t nColsRhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		MemoryTile shiftedLhsBuffer = this->_buffer;
		MemoryTile shiftedRhsBuffer = rhs._buffer;
		MemoryTile shiftedOutBuffer = out._buffer;
		if (rhsOperation == MatrixOperation::Transpose)
		{
			shiftedRhsBuffer.nRows = rhs.nCols();
			shiftedRhsBuffer.nCols = rhs.nRows();
		}
		
		if (lhsOperation == MatrixOperation::None)
			assert(this->nCols() == shiftedRhsBuffer.nRows);
		else
		{
			shiftedLhsBuffer.nRows = this->nCols();
			shiftedLhsBuffer.nCols = this->nRows();
			assert(this->nRows() == shiftedRhsBuffer.nRows);
		}
		
		assert(rowStart <= (lhsOperation == MatrixOperation::None ? this->nRows() : this->nCols()));
		assert(nRows + rowStart <= (lhsOperation == MatrixOperation::None ? this->nRows() : this->nCols()));
		assert(colStart <= (lhsOperation == MatrixOperation::None ? this->nCols() : this->nRows()));
		assert(nCols + colStart <= (lhsOperation == MatrixOperation::None ? this->nCols() : this->nRows()));
		assert(colRhsStart <= (rhsOperation == MatrixOperation::None ? rhs.nCols() : rhs.nRows()));
		assert(colRhsStart + nColsRhs <=  (rhsOperation == MatrixOperation::None ? rhs.nCols() : rhs.nRows()));
		
		size_t rowOffset;
		size_t colOffset;
		size_t rowRhsOffset;
		size_t colRhsOffset;
		if (lhsOperation == MatrixOperation::None)
		{
			rowOffset = rowStart * this->_buffer.ElementarySize();
			colOffset = colStart * this->nRows() * this->_buffer.ElementarySize();
			rowRhsOffset = 0 * rhs.GetBuffer().ElementarySize();
			colRhsOffset = colRhsStart * rhs.nRows() * rhs.GetBuffer().ElementarySize();
		}
		else
		{
			rowOffset = rowStart * this->nCols() * this->_buffer.ElementarySize();
			colOffset = colStart * this->_buffer.ElementarySize();
			rowRhsOffset = 0 * rhs.nCols() * rhs.GetBuffer().ElementarySize();
			colRhsOffset = colRhsStart * rhs.GetBuffer().ElementarySize();
		}
		
		shiftedOutBuffer.pointer += rowOffset + colRhsOffset;
		shiftedLhsBuffer.pointer += rowOffset + colOffset;
		shiftedRhsBuffer.pointer += rowRhsOffset + colRhsOffset;

		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::SubMultiply(shiftedOutBuffer, shiftedLhsBuffer, shiftedRhsBuffer, static_cast<unsigned>(nRows), static_cast<unsigned>(nCols), static_cast<unsigned>(nColsRhs), lhsOperation, rhsOperation, alpha, beta);
		else
			routines::SubMultiply(shiftedOutBuffer, shiftedLhsBuffer, shiftedRhsBuffer, static_cast<unsigned>(nRows), static_cast<unsigned>(nCols), static_cast<unsigned>(nColsRhs), lhsOperation, rhsOperation, alpha, beta);
	}

template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::Dot(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha, const double beta) const
	{
		Vector<ms, md> ret(rhs.size());
		Dot(ret, rhs, lhsOperation, alpha, beta);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Dot(Vector<ms, md>& out, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha, const double beta) const
	{
		if (lhsOperation == MatrixOperation::None)
		{
			assert(nCols() == rhs.size());
			assert(nRows() == out.size());
		}
		else
		{
			assert(nRows() == rhs.size());
			assert(nCols() == out.size());
		}
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Dot(out.GetBuffer(), this->_buffer, rhs.GetBuffer(), lhsOperation, alpha, beta);
		else
			routines::Dot(out.GetBuffer(), this->_buffer, rhs.GetBuffer(), lhsOperation, alpha, beta);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::KroneckerProduct(const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha)
	{
		ColumnWiseMatrix ret(lhs.size(), rhs.size(), 0.0);
		KroneckerProduct(ret, lhs, rhs, alpha);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::KroneckerProduct(ColumnWiseMatrix<ms, md>& out, const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::KroneckerProduct(out._buffer, lhs.GetBuffer(), rhs.GetBuffer(), alpha);
		else
			routines::KroneckerProduct(out._buffer, lhs.GetBuffer(), rhs.GetBuffer(), alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::RowWiseSum() const
	{
		Vector<ms, md> out(nRows());
		RowWiseSum(out);
		
		return out;
	}
	
	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::RowWiseSum(Vector<ms, md>& out) const
	{
		Vector<ms, md> cache(nCols(), 1.0);
		RowWiseSum(out, cache);
	}
	
	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::RowWiseSum(Vector<ms, md>& out, Vector<ms, md>& cache) const
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::RowWiseSum(out.GetBuffer(), this->_buffer, cache.GetBuffer());
		else
			routines::RowWiseSum(out.GetBuffer(), this->_buffer, cache.GetBuffer());
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::ColumnWiseSum() const
	{
		Vector<ms, md> out(nCols());
		ColumnWiseSum(out);
		
		return out;
	}
	
	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ColumnWiseSum(Vector<ms, md>& out) const
	{
		Vector<ms, md> cache(nRows(), 1.0);
		ColumnWiseSum(out, cache);
	}
	
	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ColumnWiseSum(Vector<ms, md>& out, Vector<ms, md>& cache) const
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::RowWiseSum(out.GetBuffer(), this->_buffer, cache.GetBuffer(), MatrixOperation::Transpose);
		else
			routines::RowWiseSum(out.GetBuffer(), this->_buffer, cache.GetBuffer(), MatrixOperation::Transpose);
	}

template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Invert(const MatrixOperation lhsOperation)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Invert(this->_buffer, lhsOperation);
		else
			routines::Invert(this->_buffer, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Solve(ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Solve(this->_buffer, rhs._buffer, lhsOperation);
		else
			routines::Solve(this->_buffer, rhs._buffer, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Solve(Vector<ms, md>& rhs, const MatrixOperation lhsOperation) const
	{
		assert(nRows() == rhs.size());

		MemoryTile tmp(rhs.GetBuffer());
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::Solve(this->buffer, tmp, lhsOperation);
		else
			routines::Solve(this->buffer, tmp, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ColumnWiseArgAbsMinimum(Vector<ms, MathDomain::Int>& out) const
	{
		assert(out.GetBuffer().pointer != 0);
		assert(out.size() == nCols());

		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::ColumnWiseArgAbsMin(out.GetBuffer(), _buffer);
		else
			routines::ColumnWiseArgAbsMin(out.GetBuffer(), _buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, MathDomain::Int> ColumnWiseMatrix<ms, md>::ColumnWiseArgAbsMinimum() const
	{
		Vector<ms, MathDomain::Int> out(nCols());
		ColumnWiseArgAbsMinimum(out);

		return out;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ColumnWiseArgAbsMaximum(Vector<ms, MathDomain::Int>& out) const
	{
		assert(out.GetBuffer().pointer != 0);
		assert(out.size() == nCols());

		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::ColumnWiseArgAbsMax(out.GetBuffer(), _buffer);
		else
			routines::ColumnWiseArgAbsMax(out.GetBuffer(), _buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, MathDomain::Int> ColumnWiseMatrix<ms, md>::ColumnWiseArgAbsMaximum() const
	{
		Vector<ms, MathDomain::Int> out(nCols());
		ColumnWiseArgAbsMaximum(out);

		return out;
	}


	#pragma endregion

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Copy(const ColumnWiseMatrix<ms, md>& source)
	{
		ColumnWiseMatrix<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Eye(const unsigned nRows)
	{
		ColumnWiseMatrix<ms, md> I(nRows, nRows);
		I.MakeIdentity();

		return I;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::LinSpace(const stdType x0, const stdType x1, const unsigned nRows, const unsigned nCols)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::RandomShuffleColumns(ColumnWiseMatrix<ms, md>& m, const unsigned seed)
	{
		m.RandomShuffleColumns(seed);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::RandomShuffleColumnsPair(ColumnWiseMatrix<ms, md>& m1, ColumnWiseMatrix<ms, md>& m2, const unsigned seed)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::RandShuffleColumnsPair(m1.GetTile(), m2.GetTile(), seed);
		else
			routines::RandShuffleColumnsPair(m1.GetTile(), m2.GetTile(), seed);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Print(const ColumnWiseMatrix<ms, md>& m, const std::string& label)
	{
		m.Print(label);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& ColumnWiseMatrix<ms, md>::MatrixToOutputStream(const ColumnWiseMatrix<ms, md>& m, std::ostream& os)
	{
		cl::MatrixToOutputStream(m.Get(), m.nRows(), m.nCols(), os);

		return os;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::MatrixFromInputStream(std::istream& is)
	{
		std::vector<typename ColumnWiseMatrix<ms, md>::stdType> _mat;
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromInputStream(_mat, nRows, nCols, is);
		
		ColumnWiseMatrix<ms, md> m(nRows, nCols);
		m.ReadFrom(_mat);

		return m;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::MatrixFromBinaryFile(const std::string& fileName, const bool transposed, const bool compressed, const bool useMemoryMapping)
	{
		std::vector<typename Vector<ms, md>::stdType> _mat {};
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromBinaryFile(_mat, nRows, nCols, fileName, transposed, compressed, useMemoryMapping);
		//std::swap(nRows, nCols);

		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.ReadFrom(_mat);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::MatrixToBinaryFile(const ColumnWiseMatrix<ms, md>& m, const std::string& fileName, const bool tranposed, const bool compressed, const std::string mode)
	{
		const auto& _mat = m.Get();
		cl::MatrixToBinaryFile(_mat, m.nRows(), m.nCols(), fileName, tranposed, compressed, mode);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Add(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Scale(ColumnWiseMatrix<ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Multiply(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha)
	{
		return lhs.Multiply(rhs, lhsOperation, rhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha)
	{
		lhs.Multiply(out, rhs, lhsOperation, rhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::Dot(const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		return lhs.Multiply(rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Dot(Vector<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		lhs.Multiply(out, rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, MathDomain::Float> ColumnWiseMatrix<ms, md>::MakeTriple(const Vector<ms, md>& x, const Vector<ms, md>& y, const Vector<ms, md>& z)
	{
		assert(x.size() == z.nRows());
		assert(y.size() == z.nCols());
		Vector<ms, MathDomain::Float> triple(3 * x.size() * y.size());
		MakeTriple(triple, x, y, z);

		return triple;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::MakeTriple(Vector<ms, MathDomain::Float>& triple, const Vector<ms, md>& x, const Vector<ms, md>& y, const Vector<ms, md>& z)
	{
		if (ms == MemorySpace::Host || ms == MemorySpace::Device)
			dm::detail::MakeTriple(triple.GetBuffer(), x.GetBuffer(), y.GetBuffer(), z.GetBuffer());
		else
			routines::MakeTriple(triple.GetBuffer(), x.GetBuffer(), y.GetBuffer(), z.GetBuffer());
	}
}
