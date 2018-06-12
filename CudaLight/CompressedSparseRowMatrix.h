#pragma once

#include <string>

#include <IBuffer.h>
#include <Types.h>
#include <Vector.h>
#include <ColumnWiseMatrix.h>

namespace cl
{
	/**
	 * CSR Matrix implementation
	 */
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class CompressedSparseRowMatrix : public IBuffer<CompressedSparseRowMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class IBuffer<CompressedSparseRowMatrix, memorySpace, mathDomain>;

		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows, const stdType value);
		// copy denseVector to host, numerically finds the non-zero indices, and then copy back to device
		CompressedSparseRowMatrix(const ColumnWiseMatrix<memorySpace, mathDomain>& denseMatrix);
		CompressedSparseRowMatrix(const CompressedSparseRowMatrix& rhs);

		virtual ~CompressedSparseRowMatrix() = default;

		const MemoryBuffer& GetBuffer() const noexcept override { return values.GetBuffer(); }

		std::vector<typename Traits<mathDomain>::stdType> Get() const override;
		void Print(const std::string& label = "") const override;
		std::ostream& Serialize(std::ostream& os) const override { throw std::exception("Not Implemented"); };

		unsigned denseSize() const noexcept { return nRows() * nCols(); }  // used only when converting to dense
		unsigned nRows() const noexcept { return buffer.nRows; }
		unsigned nCols() const noexcept { return buffer.nCols; }

		#pragma region Linear Algebra

		ColumnWiseMatrix<memorySpace, mathDomain> operator *(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs) const;
		Vector<memorySpace, mathDomain> operator *(const Vector<memorySpace, mathDomain>& rhs) const;
		
		ColumnWiseMatrix<memorySpace, mathDomain> Multiply(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		void Multiply(ColumnWiseMatrix<memorySpace, mathDomain>& out, const ColumnWiseMatrix<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		
		Vector<memorySpace, mathDomain> Dot(const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		void Dot(Vector<memorySpace, mathDomain>& out, const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;


		#pragma endregion 

	protected:
		SparseMemoryTile buffer;

		Vector<memorySpace, mathDomain> values;
		Vector<memorySpace, MathDomain::Int> nonZeroColumnIndices;
		Vector<memorySpace, MathDomain::Int> nNonZeroRows;

		using IBuffer<CompressedSparseRowMatrix, memorySpace, mathDomain>::Alloc;

	private:
		/**
		* buffer.pointer <- values.pointer
		* buffer.nonZeroColumnIndices <- nonZeroColumnIndices.pointer
		* buffer.nNonZeroRows <- nNonZeroRows.pointer
		*/
		void SyncPointers();
	};

	#pragma region Type aliases

	typedef CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Int> GpuIntegerSparseMatrix;
	typedef CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Float> GpuSingleSparseMatrix;
	typedef GpuSingleSparseMatrix GpuFloatSparseMatrix;
	typedef CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Double> GpuDoubleSparseMatrix;

	typedef CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Int> CpuIntegerSparseMatrix;
	typedef CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Float> CpuSingleSparseMatrix;
	typedef CpuSingleSparseMatrix CpuFloatSparseMatrix;
	typedef CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Double> CpuDoubleSparseMatrix;

	typedef GpuSingleSparseMatrix smat;
	typedef GpuDoubleSparseMatrix dsmat;
	typedef GpuIntegerSparseMatrix ismat;

	#pragma endregion

	#pragma region

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Multiply(const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	* Same version as above, but gives the possibility of reusing the output buffer
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Dot(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	* Same version as above, but gives the possibility of reusing the output buffer
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Dot(Vector<ms, md>& out, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);

	#pragma endregion 
}

namespace cl
{
	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<ms, MathDomain::Int>& nonZeroColumnIndices, const Vector<ms, MathDomain::Int>& nNonZeroRows)
		: IBuffer(false),  // CompressedSparseRowMatrix doesn't allocate its memory in its buffer!
		  buffer(0, nonZeroColumnIndices.size(), 0, 0, nRows, nCols, ms, md),
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
		: IBuffer(false), buffer(0, 0, 0, 0, denseMatrix.nRows(), denseMatrix.nCols(), ms, md)
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

		buffer.size = nNonZeros;

		values.buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroValues.size()), ms, md);
		Alloc(values.buffer);
		values.ReadFrom(nonZeroValues);

		this->nonZeroColumnIndices.buffer = MemoryBuffer(0, static_cast<unsigned>(nonZeroColumnIndices.size()), ms, md);
		Alloc(this->nonZeroColumnIndices.buffer);
		this->nonZeroColumnIndices.ReadFrom(nonZeroColumnIndices);

		this->nNonZeroRows.buffer = MemoryBuffer(0, static_cast<unsigned>(nNonZeroRows.size()), ms, md);
		Alloc(this->nNonZeroRows.buffer);
		this->nNonZeroRows.ReadFrom(nNonZeroRows);

		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	CompressedSparseRowMatrix<ms, md>::CompressedSparseRowMatrix(const CompressedSparseRowMatrix& rhs)
		: IBuffer(false), // CompressedSparseRowMatrix doesn't allocate its memory in its buffer!
		  buffer(0, 0, 0, 0, rhs.nRows(), rhs.nCols(), ms, md),
		  values(rhs.values), nonZeroColumnIndices(rhs.nonZeroColumnIndices), nNonZeroRows(rhs.nNonZeroRows)
	{
		SyncPointers();
	}

	template< MemorySpace ms, MathDomain md>
	void CompressedSparseRowMatrix<ms, md>::SyncPointers()
	{
		buffer.pointer = values.buffer.pointer;
		buffer.nonZeroColumnIndices = nonZeroColumnIndices.buffer.pointer;
		buffer.nNonZeroRows = nNonZeroRows.buffer.pointer;
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
		dm::detail::SparseMultiply(ret.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows());

		return ret;
	}

	template< MemorySpace ms, MathDomain md>
	Vector<ms, md> CompressedSparseRowMatrix<ms, md>::operator *(const Vector<ms, md>& rhs) const
	{
		assert(nRows() == rhs.size());

		Vector<ms, md> ret(rhs.size());
		dm::detail::SparseDot(ret.buffer, this->buffer, rhs.buffer);

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
		dm::detail::SparseMultiply(out.buffer, this->buffer, rhs.buffer, this->nRows(), rhs.nRows(), lhsOperation, alpha);
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
		dm::detail::SparseDot(out.buffer, this->buffer, rhs.buffer, lhsOperation, 1.0);
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