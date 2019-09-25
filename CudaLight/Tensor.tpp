#pragma once

#include <CompressedSparseRowMatrix.h>

namespace cl
{
	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices)
		: IBuffer<Tensor<ms, md>, ms, md>(true), _buffer(MemoryCube(0, nRows, nCols, nMatrices, ms, md))
	{
		this->ctor(_buffer);

		matrices.resize(nMatrices);
		for (size_t i = 0; i < nMatrices; i++)
		{
			const size_t matrixShift = i * nRows * nCols * _buffer.ElementarySize();
			MemoryTile matrixBuffer(_buffer.pointer + matrixShift, _buffer.nRows, _buffer.nCols, ms, md);
			matrices[i] = ColumnWiseMatrix<ms, md>::make_shared(matrixBuffer);

			matrices[i]->columns.resize(nCols);
			for (size_t j = 0; j < nCols; ++j)
			{
				const size_t colShift = j * nRows * _buffer.ElementarySize();
				MemoryBuffer colBuffer(_buffer.pointer + colShift, _buffer.nRows, ms, md);
				matrices[i]->columns[j] = Vector<ms, md>::make_shared(colBuffer);
			}
		}
	}
	
	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(Tensor&& rhs) noexcept
			: IBuffer<Tensor<ms, md>, ms, md>(std::move(rhs)), matrices(std::move(rhs.matrices)), _buffer(rhs._buffer)
	{
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const typename Traits<md>::stdType value)
		: Tensor(nRows, nCols, nMatrices)
	{
		dm::detail::Initialize(_buffer, static_cast<double>(value));
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const unsigned nRows, const unsigned nMatrices)
		: ColumnWiseMatrix<ms, md>(nRows, nRows, nMatrices)
	{
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const Tensor& rhs)
		: Tensor(rhs.nRows(), rhs.nCols(), rhs.nMatrices())
	{
		ReadFrom(rhs);
	}

	template<MemorySpace ms, MathDomain md>
	template<typename T>
	Tensor<ms, md>::Tensor(const std::vector<T>& rhs, const unsigned nRows, const unsigned nCols, const unsigned nMatrices)
		: Tensor(nRows, nCols, nMatrices)
	{
		assert(rhs.size() == nRows * nCols * nMatrices);
		ReadFrom(rhs);
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
		dm::detail::AutoCopy(matrices[0]->columns[0]->GetBuffer(), rhs.GetBuffer());
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md>::Tensor(const MemoryCube& buffer)
		: IBuffer<Tensor<ms, md>, ms, md>(false), _buffer(buffer)
	{

	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::ReadFrom(const ColumnWiseMatrix<ms, md>& rhs)
	{
		assert(_buffer.pointer != 0);
		assert(rhs._buffer.pointer != 0);
		assert(rhs.size() != 0);

		dm::detail::AutoCopy(static_cast<MemoryBuffer>(matrices[0]->_buffer), rhs._buffer);
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::ReadFrom(const Vector<ms, md>& rhs)
	{
		assert(_buffer.pointer != 0);
		assert(rhs._buffer.pointer != 0);
		assert(rhs.size() != 0);

		dm::detail::AutoCopy(matrices[0]->columns[0]->_buffer, rhs._buffer);
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> Tensor<ms, md>::Get(const unsigned matrix) const
	{
		assert(matrix < nMatrices());
		return matrices[matrix]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	std::vector<typename Traits<md>::stdType> Tensor<ms, md>::Get(const unsigned matrix, const unsigned column) const
	{
		assert(matrix < nMatrices());
		assert(column < nCols());
		return matrices[matrix]->columns[column]->Get();
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::Set(const ColumnWiseMatrix<ms, md>& matrixBuffer, const unsigned matrix)
	{
		assert(matrix < nMatrices());
		assert(matrixBuffer._buffer.pointer != 0);
		matrices[matrix]->ReadFrom(matrixBuffer);
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::Set(const Vector<ms, md>& columnVector, const unsigned column, const unsigned matrix)
	{
		assert(matrix < nMatrices());
		assert(column < nCols());
		assert(columnVector.GetBuffer().pointer != 0);
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

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Tensor<ms, md>::CubeWiseSum() const
	{
		ColumnWiseMatrix<ms, md> out(nRows(), nCols(), 0.0);
		CubeWiseSum(out);
		
		return out;
	}

	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::CubeWiseSum(ColumnWiseMatrix<ms, md>& out) const
	{
		#define RUN_MULTIPLE_ADD
		#ifdef RUN_MULTIPLE_ADD
			for (size_t k = 0; k < nMatrices(); ++k)
				out.AddEqualMatrix(*this->matrices[k]);
		#else
			Vector<ms, MathDomain::Int> nNonZeroRows(out.size() * nMatrices());
			nNonZeroRows.LinSpace(0, static_cast<int>(out.size() * nMatrices() - 1));
			nNonZeroRows.Scale(nMatrices());
			
			std::vector<int> nonZeroColumnIndicesCpu(nNonZeroRows.size());
			for (size_t i = 0; i < out.size(); ++i)
			{
				for (size_t k = 0; k < nMatrices(); ++k)
					nonZeroColumnIndicesCpu[k + nMatrices() * i] = static_cast<int>(k * out.size() + i);
			}
			Vector<ms, MathDomain::Int> nonZeroColumnIndices(nonZeroColumnIndicesCpu);
			CompressedSparseRowMatrix<ms, md> eye(out.size(), out.size() * nMatrices(), nonZeroColumnIndices, nNonZeroRows, 1.0);
			CubeWiseSum(out, eye);
		#endif
	}
	
	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::CubeWiseSum(ColumnWiseMatrix<ms, md>& out, const CompressedSparseRowMatrix<ms, md>&) const
	{
		#ifdef RUN_MULTIPLE_ADD
			for (size_t k = 0; k < nMatrices(); ++k)
				out.AddEqualMatrix(*this->matrices[k]);
		#else
			assert(out.size() == cache.nRows());
			assert(cache.nCols() == out.size() * nMatrices());
			assert(cache.size() == out.size() * nMatrices());
			dm::detail::SparseDot(out.GetBuffer(), cache.GetCsrBuffer(), _buffer, MatrixOperation::None, 1.0, 1.0);
		#endif
	}

	template<MemorySpace ms, MathDomain md>
	ColumnWiseMatrix<ms, md> Tensor<ms, md>::MatrixSum() const
	{
		ColumnWiseMatrix<ms, md> out(nRows(), nCols(), -123456789.0);
		MatrixSum(out);
		
		return out;
	}
	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::MatrixSum(ColumnWiseMatrix<ms, md>& out) const
	{
		Vector<ms, md> cacheOnes(nMatrices(), 1.0);
		MatrixSum(out, cacheOnes);
	}
	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::MatrixSum(ColumnWiseMatrix<ms, md>& out, Vector<ms, md>& cacheOnes) const
	{
		MemoryCube tmp1(out.GetBuffer().pointer, out.nRows(), 1, nMatrices(), _buffer.memorySpace, _buffer.mathDomain);
		MemoryCube tmp2(_buffer.pointer, _buffer.nRows, _buffer.nCols, 0, _buffer.memorySpace, _buffer.mathDomain);
		MemoryCube tmp3(cacheOnes.GetBuffer().pointer, cacheOnes.size(), 0, 0, _buffer.memorySpace, _buffer.mathDomain);
		dm::detail::BatchedMultiply(tmp1, tmp2, tmp3,
									tmp2.nRows * tmp2.nCols,0,
									MatrixOperation::None, MatrixOperation::None, 1.0, 0.0);
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Tensor<ms, md>::KroneckerProduct(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const double alpha)
	{
		Tensor<ms, md> ret(lhs.nRows(), rhs.nRows(), rhs.nCols(), 0.0);
		KroneckerProduct(ret, lhs, rhs, alpha);
		
		return ret;
	}
	
	template<MemorySpace ms, MathDomain md>
	void Tensor<ms, md>::KroneckerProduct(Tensor<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const double alpha)
	{
		assert(out.nMatrices() == lhs.nCols());
		assert(out.nMatrices() == rhs.nCols());
		
		//#define DO_NOT_USE_STREAMS
		#ifdef DO_NOT_USE_STREAMS
			for (size_t k = 0; k < out.nMatrices(); ++k)
				dm::detail::KroneckerProduct(out.matrices[k]->GetTile(), lhs.columns[k]->GetBuffer(), rhs.columns[k]->GetBuffer(), alpha);
		#else
			dm::detail::BatchedTransposedKroneckerProduct(out.GetCube(), lhs.GetTile(), rhs.GetTile(), alpha);
		#endif
		
	}
	
#pragma endregion

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> Copy(const Tensor<ms, md>& source)
	{
		Tensor<ms, md> ret(source);
		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned nRows, const unsigned nCols, const unsigned nMatrices)
	{
		Tensor<ms, md> ret(nRows, nCols, nMatrices);
		ret.LinSpace(x0, x1);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
	Tensor<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed)
	{
		Tensor<ms, md> ret(nRows, nCols, nMatrices);
		ret.RandomUniform(seed);

		return ret;
	}

	template<MemorySpace ms, MathDomain md>
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
