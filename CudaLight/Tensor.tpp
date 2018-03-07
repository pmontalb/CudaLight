#pragma once

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices)
		: IBuffer(true), buffer(MemoryCube(0, nRows, nCols, nMatrices, ms, md))
	{
		ctor(buffer);

		matrices.resize(nCols);
		for (size_t i = 0; i < nCols; i++)
		{
			const size_t matrixShift = i * nRows * nCols * buffer.ElementarySize();
			MemoryTile matrixBuffer(buffer.pointer + matrixShift, buffer.nRows, buffer.nCols, ms, md);
			matrices[i] = ColumnWiseMatrix<ms, md>::make_shared(matrixBuffer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const double value)
		: Tensor(nRows, nCols, nMatrices)
	{
		dm::detail::Initialize(static_cast<MemoryBuffer>(buffer), value);
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nMatrices)
		: ColumnWiseMatrix(nRows, nRows, nMatrices)
	{
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const Tensor& rhs)
		: Tensor(rhs.nRows(), rhs.nCols(), rhs.nMatrices())
	{
		dm::detail::AutoCopy(static_cast<MemoryBuffer>(buffer), static_cast<MemoryBuffer>(rhs.buffer));
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const ColumnWiseMatrix<ms, md>& rhs)
		: Tensor(rhs.nRows(), rhs.nCols(), 1)
	{
		dm::detail::AutoCopy(matrices[0]->GetBuffer(), static_cast<MemoryBuffer>(rhs.GetBuffer()));
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const Vector<ms, md>& rhs)
		: Tensor(rhs.size(), 1, 1)
	{
		dm::detail::AutoCopy(matrices[0]->columns[0]->buffer, rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const MemoryCube& buffer)
		: IBuffer(false), buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::ReadFrom(const ColumnWiseMatrix<ms, md>& rhs)
	{
		if (!buffer.pointer)
			throw BufferNotInitialisedException();

		dm::detail::AutoCopy(static_cast<MemoryBuffer>(matrices[0]->buffer), rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::ReadFrom(const Vector<ms, md>& rhs)
	{
		if (!buffer.pointer)
			throw BufferNotInitialisedException();

		dm::detail::AutoCopy(matrices[0]->columns[0]->buffer, rhs.buffer);
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<double> Tensor<ms, md>::Get(const unsigned matrix) const
	{
		return matrices[matrix]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<double> Tensor<ms, md>::Get(const unsigned matrix, const unsigned column) const
	{
		return matrices[matrix]->columns[column]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::Set(const ColumnWiseMatrix<ms, md>& matrixBuffer, const unsigned matrix)
	{
		assert(matrix < nMatrices());
		matrices[matrix]->ReadFrom(matrixBuffer);
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::Set(const Vector<ms, md>& columnVector, const unsigned column, const unsigned matrix)
	{
		assert(matrix < nMatrices());
		assert(column < nCols());
		matrices[matrix]->columns[column]->ReadFrom(columnVector);
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::Print(const std::string& label) const
	{
		auto ten = Get();

		std::cout << "********* " << label << " ***********" << std::endl;
		for (size_t k = 0; k < nCols(); k++)
		{
			std::cout << std::endl;
			for (size_t j = 0; j < nCols(); j++)
			{
				std::cout << "\t";
				for (size_t i = 0; i < nRows(); i++)
					std::cout << " t[" << i << "][" << j << "][" << k << "] = " << ten[i + nRows() * (j + nMatrices() * k)];
				std::cout << std::endl;
			}
		}
		std::cout << "**********************" << std::endl;
	}

	#pragma region Linear Algebra

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Tensor<ms, md>::operator +(const Tensor& rhs) const
	{
		Tensor ret(*this);
		ret += rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Tensor<ms, md>::operator -(const Tensor& rhs) const
	{
		Tensor ret(*this);
		ret -= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Tensor<ms, md>::operator %(const Tensor& rhs) const
	{
		Tensor ret(*this);
		ret %= rhs;

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Tensor<ms, md>::Add(const Tensor& rhs, const double alpha) const
	{
		Tensor ret(*this);
		ret.AddEqual(rhs, alpha);

		return ret;
	}

	#pragma endregion

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> Copy(const Tensor<ms, md>& source)
	{
		Tensor<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> LinSpace(const double x0, const double x1, const unsigned nRows, const unsigned nCols, const unsigned nMatrices)
	{
		Tensor<ms, md> ret(nRows, nCols, nMatrices);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed)
	{
		Tensor<ms, md> ret(nRows, nCols, nMatrices);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed)
	{
		Tensor<ms, md> ret(nRows, nCols, nMatrices);
		ret.RandomGaussian(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	void Print(const Tensor<ms, md>& ten, const std::string& label)
	{
		ten.Print(label);
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Add(const Tensor<ms, md>& lhs, const Tensor<ms, md>& rhs, const double alpha)
	{
		return lhs.Add(rhs, alpha);
	}

	template<MemorySpace ms, MathDomain md>
	void Scale(Tensor<ms, md>& lhs, const double alpha)
	{
		lhs.Scale(alpha);
	}
}