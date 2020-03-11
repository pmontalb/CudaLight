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
		
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<memorySpace, MathDomain::Int>&& nonZeroColumnIndices, Vector<memorySpace, MathDomain::Int>&& nNonZeroRows);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<memorySpace, MathDomain::Int>&& nonZeroColumnIndices, Vector<memorySpace, MathDomain::Int>&& nNonZeroRows, const stdType value);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows, const stdType value);
		// copy denseVector to host, numerically finds the non-zero indices, and then copy back to device
		explicit CompressedSparseRowMatrix(const ColumnWiseMatrix<memorySpace, mathDomain>& denseMatrix);
		CompressedSparseRowMatrix(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols);
		explicit CompressedSparseRowMatrix(const CompressedSparseRowMatrix& rhs);
		explicit CompressedSparseRowMatrix(CompressedSparseRowMatrix&& rhs) noexcept;

		void ReadFrom(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols);

		inline ~CompressedSparseRowMatrix() override
        {
            this->dtor(_buffer);
            _buffer.pointer = 0;
        }

		const MemoryBuffer& GetBuffer() const noexcept override final { return values.GetBuffer(); }
		MemoryBuffer& GetBuffer() noexcept override final { return values.GetBuffer(); }
		
		const SparseMemoryTile& GetCsrBuffer() const noexcept { return _buffer; }
		SparseMemoryTile& GetCsrBuffer() noexcept { return _buffer; }

		std::vector<typename Traits<mathDomain>::stdType> Get() const override final;
		void Print(const std::string& label = "") const override final;
		void PrintNonZeros(const std::string& label = "") const;
		std::ostream& ToOutputStream(std::ostream&) const override final { throw std::logic_error("Not Implemented"); }
		void ToBinaryFile(const std::string&, const bool, const std::string) const override final { throw std::logic_error("Not Implemented"); }

		unsigned denseSize() const noexcept { return nRows() * nCols(); }  // used only when converting to dense
		unsigned nRows() const noexcept { return _buffer.nRows; }
		unsigned nCols() const noexcept { return _buffer.nCols; }

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
		SparseMemoryTile _buffer = SparseMemoryTile(0, 0, 0, 0, 0, 0, memorySpace, mathDomain);

		Vector<memorySpace, mathDomain> values {};
		Vector<memorySpace, MathDomain::Int> nonZeroColumnIndices {};
		Vector<memorySpace, MathDomain::Int> nNonZeroRows {};

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

#include <CompressedSparseRowMatrix.tpp>
