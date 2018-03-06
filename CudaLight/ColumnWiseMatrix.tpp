#pragma once

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols)
		: IBuffer(true), buffer(MemoryTile(0, nRows, nCols, ms, md))
	{
		ctor(buffer);

		columns.resize(nCols);
		for (size_t i = 0; i < nCols; i++)
		{
			const size_t colShift = i * nRows * buffer.ElementarySize();
			MemoryBuffer colBuffer(buffer.pointer + colShift, buffer.nRows, ms, md);
			columns[i] = Vector<ms, md>::make_shared(colBuffer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const unsigned nRows, const unsigned nCols, const double value)
		: ColumnWiseMatrix(nRows, nCols)
	{
		dm::detail::Initialize(static_cast<MemoryBuffer>(buffer), value);
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
		dm::detail::AutoCopy(static_cast<MemoryBuffer>(buffer), static_cast<MemoryBuffer>(rhs.buffer));
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md>::ColumnWiseMatrix(const Vector<ms, md>& rhs)
		: ColumnWiseMatrix(rhs.size(), 1)
	{
		dm::detail::AutoCopy(static_cast<MemoryBuffer>(columns[0]->buffer), rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::ReadFrom(const Vector<ms, md>& rhs)
	{
		if (!buffer.pointer)
			throw std::exception("Buffer needs to be initialised first!");

		dm::detail::AutoCopy(columns[0]->buffer, rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<double> ColumnWiseMatrix<ms, md>::Get(const unsigned column) const
	{
		return columns[column]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Set(const Vector<ms, md>& columnVector, const unsigned column)
	{
		assert(column < nCols());
		columns[column]->ReadFrom(columnVector);
	}

	template<MemorySpace ms, MathDomain md>
	void ColumnWiseMatrix<ms, md>::Print(const std::string& label) const
	{
		auto mat = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t j = 0; j < nCols(); j++)
		{
			std::cout << "\t";
			for (size_t i = 0; i < nRows(); i++)
				std::cout << " v[" << i << "][" << j << "] = " << mat[i + nRows() * j];
			std::cout << std::endl;
		}
		std::cout << "**********************" << std::endl;
	}

	#pragma region Linear Algebra

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator +(const ColumnWiseMatrix& rhs) const
	{
		ColumnWiseMatrix ret(*this);
		ret += rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator -(const ColumnWiseMatrix& rhs) const
	{
		ColumnWiseMatrix ret(*this);
		ret -= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator %(const ColumnWiseMatrix& rhs) const
	{
		ColumnWiseMatrix ret(*this);
		ret %= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Add(const ColumnWiseMatrix& rhs, const double alpha) const
	{
		ColumnWiseMatrix ret(*this);
		ret.AddEqual(rhs, alpha);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *(const ColumnWiseMatrix& rhs) const
	{
		ColumnWiseMatrix ret(*this);
		dm::detail::Multiply(ret.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::operator *=(const ColumnWiseMatrix& rhs) const
	{
		ColumnWiseMatrix ret(*this);
		dm::detail::Multiply(ret.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows());

		dm::detail::AutoCopy(buffer, ret.buffer);
		return *this;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::operator *(const Vector<ms, md>& rhs) const
	{
		Vector<ms, md> ret(rhs.size());
		dm::detail::Dot(ret.GetBuffer(), this->buffer, rhs.GetBuffer());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> ColumnWiseMatrix<ms, md>::Multiply(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation, const MatrixOperation rhsOperation, const double alpha) const
	{
		ColumnWiseMatrix ret(*this);
		dm::detail::Multiply(ret.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows());

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Vector<ms, md> ColumnWiseMatrix<ms, md>::Dot(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation, const double alpha) const
	{
		Vector<ms, md> ret(rhs.size());
		dm::detail::Dot(ret.GetBuffer(), this->buffer, rhs.GetBuffer(), lhsOperation, alpha);
	}

	#pragma endregion

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Copy(const ColumnWiseMatrix<ms, md>& source)
	{
		ColumnWiseMatrix<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> LinSpace(const double x0, const double x1, const unsigned nRows, const unsigned nCols)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned seed)
	{
		ColumnWiseMatrix<ms, md> ret(nRows, nCols);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void Print(const ColumnWiseMatrix<ms, md>& mat, const std::string& label)
	{
		mat.Print(label);
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
}