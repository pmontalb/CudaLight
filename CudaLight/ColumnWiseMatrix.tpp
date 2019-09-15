#pragma once

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols)
		: IBuffer<ColumnWiseMatrix<ms, md>, ms, md>(true), _buffer(MemoryTile(0, nRows, nCols, ms, md))
	{
		this->ctor(_buffer);

		columns.resize(nCols);
		for (size_t i = 0; i < nCols; i++)
		{
			const size_t colShift = i * nRows * _buffer.ElementarySize();
			MemoryBuffer colBuffer(_buffer.pointer + colShift, _buffer.nRows, ms, md);
			columns[i] = Vector<ms, md>::make_shared(colBuffer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols, const typename Traits<md>::stdType value)
		: ColumnWiseMatrix(nRows, nCols)
	{
		dm::detail::Initialize(static_cast<MemoryBuffer>(_buffer), static_cast<double>(value));
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
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const std::string& fileName, bool useMemoryMapping)
	{
		std::vector<typename Traits<md>::stdType> mat {};
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromBinaryFile(mat, nRows, nCols, fileName, useMemoryMapping);
		
		ReadFrom(mat, nRows, nCols);
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
		dm::detail::AutoCopy(columns[0]->GetBuffer(), rhs.GetBuffer());
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const MemoryTile& buffer)
		: IBuffer<ColumnWiseMatrix<ms, md>, ms, md>(false), _buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::MakeIdentity()
	{
		assert(nRows() == nCols());
		dm::detail::Eye(this->_buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::Flatten() const
	{
		Vector<ms, md> ret(this->size());
		dm::detail::AutoCopy(ret.GetBuffer(), _buffer);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ReadFrom(const Vector<ms, md>& rhs)
	{
		assert(rhs.size() > 0);
		assert(rhs.GetBuffer().pointer != 0);
		assert(_buffer.pointer != 0);

		dm::detail::AutoCopy(columns[0]->buffer, rhs.GetBuffer());
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
	void ColumnWiseMatrix<ms, md>::RandomShuffleColumns(const unsigned seed) const
	{
		dm::detail::RandShuffleColumns(this->_buffer, seed);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Print(const std::string& label) const
	{
		auto mat = Get();
		cl::Print(mat, nRows(), nCols(), label);
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
		cl::MatrixToBinaryFile(Get(), nRows(), nCols(), fileName, compressed, mode);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& operator<<(std::ostream& os, const ColumnWiseMatrix<ms, md>& buffer)
	{
		buffer.ToOutputStream(os);
		return os;
	}

	#pragma region Linear Algebra
	
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
		dm::detail::AddEqualMatrix(ret._buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *(const ColumnWiseMatrix& rhs) const
	{
		assert(nCols() == rhs.nRows());

		ColumnWiseMatrix ret(nRows(), rhs.nCols());
		dm::detail::Multiply(ret._buffer, this->_buffer, rhs._buffer, this->nRows(), rhs.nRows());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *=(const ColumnWiseMatrix& rhs) const
	{
		assert(nCols() == rhs.nRows());

		ColumnWiseMatrix ret(nRows(), rhs.nCols());
		dm::detail::Multiply(ret.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows());

		dm::detail::AutoCopy(_buffer, ret.buffer);
		return *this;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::operator *(const Vector<ms, md>& rhs) const
	{
		assert(nCols() == rhs.size());

		Vector<ms, md> ret(nRows());
		dm::detail::Dot(ret.GetBuffer(), this->_buffer, rhs.GetBuffer());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>& ColumnWiseMatrix<ms, md>::AddEqualMatrix(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta)
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());
		
		assert(_buffer.pointer != 0);
		
		dm::detail::AddEqualMatrix(_buffer, rhs._buffer, lhsOperation, rhsOperation, alpha, beta);
		return *this;
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Multiply(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		ColumnWiseMatrix ret(nRows(), rhs.nCols());
		Multiply(ret, rhs, lhsOperation, rhsOperation, alpha, beta);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Multiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		this->SubMultiply(out, rhs, this->nRows(), this->nCols(), rhs.nCols(), lhsOperation, rhsOperation, alpha, beta);
	}
	
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::SubMultiply(const ColumnWiseMatrix& rhs, const size_t nRows, const size_t nCols, const size_t nColsRhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		ColumnWiseMatrix ret(this->nRows(), rhs.nCols(), 0.0);
		SubMultiply(ret, rhs, nRows, nCols, nColsRhs, lhsOperation, rhsOperation, alpha, beta);
		
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::SubMultiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const size_t nRows, const size_t nCols, const size_t nColsRhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha, const double beta) const
	{
		if (lhsOperation == MatrixOperation::None)
			assert(this->nCols() == rhs.nRows());
		else
			assert(this->nRows() == rhs.nCols());
		
		dm::detail::SubMultiply(out._buffer, this->_buffer, rhs._buffer, out.nRows(), this->nRows(), rhs.nRows(), nRows, nCols, nColsRhs, lhsOperation, rhsOperation, alpha, beta);
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
		dm::detail::Dot(out.GetBuffer(), this->_buffer, rhs.GetBuffer(), lhsOperation, alpha, beta);
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
		dm::detail::KroneckerProduct(out._buffer, lhs.GetBuffer(), rhs.GetBuffer(), alpha);
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
		dm::detail::RowWiseSum(out.GetBuffer(), this->_buffer, cache.GetBuffer());
	}

template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Invert(const MatrixOperation lhsOperation)
	{
		dm::detail::Invert(this->_buffer, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Solve(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation) const
	{
		assert(nRows() == rhs.nRows());
		assert(nCols() == rhs.nCols());
		dm::detail::Solve(this->_buffer, rhs._buffer, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Solve(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation) const
	{
		assert(nRows() == rhs.size());

		MemoryTile tmp(rhs.GetBuffer());
		dm::detail::Solve(this->buffer, tmp, lhsOperation);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ColumnWiseArgAbsMinimum(Vector<ms, MathDomain::Int>& out) const
	{
		assert(out.GetBuffer().pointer != 0);
		assert(out.size() == nCols());

		dm::detail::ColumnWiseArgAbsMin(out.GetBuffer(), _buffer);
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

		dm::detail::ColumnWiseArgAbsMax(out.GetBuffer(), _buffer);
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
	ColumnWiseMatrix<ms, md> Copy(const ColumnWiseMatrix<ms, md>& source)
	{
		ColumnWiseMatrix<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Eye(const unsigned nRows)
	{
		ColumnWiseMatrix<ms, md> I(nRows, nRows);
		I.MakeIdentity();

		return I;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned nRows, const unsigned nCols)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void RandomShuffleColumns(ColumnWiseMatrix<ms, md>& m, const unsigned seed)
	{
		m.RandomShuffleColumns(seed);
	}
	
	template<MemorySpace ms, MathDomain md>
	void RandomShuffleColumnsPair(ColumnWiseMatrix<ms, md>& m1, ColumnWiseMatrix<ms, md>& m2, const unsigned seed)
	{
		dm::detail::RandShuffleColumnsPair(m1.GetTile(), m2.GetTile(), seed);
	}
	
	template<MemorySpace ms, MathDomain md>
	void Print(const ColumnWiseMatrix<ms, md>& mat, const std::string& label)
	{
		mat.Print(label);
	}

	template<MemorySpace ms, MathDomain md>
	std::ostream& MatrixToOutputStream(const ColumnWiseMatrix<ms, md>& mat, std::ostream& os)
	{
		cl::MatrixToOutputStream(mat.Get(), mat.nRows(), mat.nCols(), os);

		return os;
	}

	template<MemorySpace ms, MathDomain md>
	void MatrixFromBinaryFile(const ColumnWiseMatrix<ms, md>& mat, const std::string& fileName, const bool compressed, const std::string mode)
	{
		const auto& _mat = mat.Get();
		cl::MatrixToBinaryFile(_mat, mat.nRows(), mat.nCols(), fileName, compressed, mode);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> MatrixFromInputStream(std::istream& is)
	{
		std::vector<typename ColumnWiseMatrix<ms, md>::stdType> _mat;
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromInputStream(_mat, nRows, nCols, is);
		
		ColumnWiseMatrix<ms, md> mat(nRows, nCols);
		mat.ReadFrom(_mat);

		return mat;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> MatrixFromBinaryFile(const std::string& fileName, const bool compressed, const bool useMemoryMapping)
	{
		std::vector<typename Vector<ms, md>::stdType> _mat {};
		unsigned nRows = 0, nCols = 0;
		cl::MatrixFromBinaryFile(_mat, nRows, nCols, fileName, compressed, useMemoryMapping);

		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.ReadFrom(_mat);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Add(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Scale(ColumnWiseMatrix<ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Multiply(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha)
	{
		return lhs.Multiply(rhs, lhsOperation, rhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha)
	{
		lhs.Multiply(out, rhs, lhsOperation, rhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> Dot(const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		return lhs.Multiply(rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Dot(Vector<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha)
	{
		lhs.Multiply(out, rhs, lhsOperation, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, MathDomain::Float> MakeTriple(const Vector<ms, md>& x, const Vector<ms, md>& y, const Vector<ms, md>& z)
	{
		assert(x.size() == z.nRows());
		assert(y.size() == z.nCols());
		Vector<ms, MathDomain::Float> triple(3 * x.size() * y.size());
		MakeTriple(triple, x, y, z);

		return triple;
	}

	template<MemorySpace ms, MathDomain md>
	void MakeTriple(Vector<ms, MathDomain::Float>& triple, const Vector<ms, md>& x, const Vector<ms, md>& y, const Vector<ms, md>& z)
	{
		dm::detail::MakeTriple(triple.GetBuffer(), x.GetBuffer(), y.GetBuffer(), z.GetBuffer());
	}
}
