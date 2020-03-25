#pragma once

#include <string>

#include <Buffer.h>
#include <Types.h>
#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <HostRoutines/SparseWrappers.h>

namespace cl
{
	/**
	 * CSR Matrix implementation
	 */
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class CompressedSparseRowMatrix : public Buffer<CompressedSparseRowMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class Buffer<CompressedSparseRowMatrix, memorySpace, mathDomain>;
		
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<memorySpace, MathDomain::Int>&& nonZeroColumnIndices, Vector<memorySpace, MathDomain::Int>&& nNonZeroRows);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, Vector<memorySpace, MathDomain::Int>&& nonZeroColumnIndices, Vector<memorySpace, MathDomain::Int>&& nNonZeroRows, const stdType value);
		CompressedSparseRowMatrix(const unsigned nRows, const unsigned nCols, const Vector<memorySpace, MathDomain::Int>& nonZeroColumnIndices, const Vector<memorySpace, MathDomain::Int>& nNonZeroRows, const stdType value);
		// copy denseVector to host, numerically finds the non-zero indices, and then copy back to device
		explicit CompressedSparseRowMatrix(const ColumnWiseMatrix<memorySpace, mathDomain>& denseMatrix);
		CompressedSparseRowMatrix(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols);
		CompressedSparseRowMatrix(const CompressedSparseRowMatrix& rhs);
		CompressedSparseRowMatrix(CompressedSparseRowMatrix&& rhs) noexcept;

		void ReadFrom(const std::vector<stdType>& denseMatrix, const size_t nRows, const size_t nCols);

		inline ~CompressedSparseRowMatrix() override
        {
            this->dtor(_buffer);
            _buffer.pointer = 0;
            
            if (_buffer.thirdPartyHandle == 0)
			{
            	switch (_buffer.memorySpace)
				{
					case MemorySpace::Device:
					case MemorySpace::Host:
						dm::detail::DestroyCsrHandle(_buffer);
						break;
					default:
						routines::DestroyCsrHandle(_buffer);
				}
			}
        }

		const MemoryBuffer& GetBuffer() const noexcept final { return values.GetBuffer(); }
		MemoryBuffer& GetBuffer() noexcept final { return values.GetBuffer(); }
		
		const SparseMemoryTile& GetCsrBuffer() const noexcept { return _buffer; }
		SparseMemoryTile& GetCsrBuffer() noexcept { return _buffer; }

		std::vector<typename Traits<mathDomain>::stdType> Get() const final;
		void Print(const std::string& label = "") const final;
		void PrintNonZeros(const std::string& label = "") const;
		std::ostream& ToOutputStream(std::ostream&) const final { throw std::logic_error("Not Implemented"); }
		void ToBinaryFile(const std::string&, const bool, const std::string) const final { throw std::logic_error("Not Implemented"); }

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

		/**
		* Solve A * X = B, B is overwritten
		*/
		void Solve(ColumnWiseMatrix<memorySpace, mathDomain>& rhs, LinearSystemSolverType solver = LinearSystemSolverType::Lu) const;

		/**
		* Solve A * x = b, b is overwritten
		*/
		void Solve(Vector<memorySpace, mathDomain>& rhs, LinearSystemSolverType solver = LinearSystemSolverType::Lu) const;


#pragma endregion

	protected:
		SparseMemoryTile _buffer = SparseMemoryTile(0, 0, 0, 0, 0, 0, memorySpace, mathDomain);

		Vector<memorySpace, mathDomain> values {};
		Vector<memorySpace, MathDomain::Int> nonZeroColumnIndices {};
		Vector<memorySpace, MathDomain::Int> nNonZeroRows {};

		using Buffer<CompressedSparseRowMatrix, memorySpace, mathDomain>::Alloc;

	private:
		/**
		* buffer.pointer <- values.pointer
		* buffer.nonZeroColumnIndices <- nonZeroColumnIndices.pointer
		* buffer.nNonZeroRows <- nNonZeroRows.pointer
		*/
		void SyncPointers();
	};
	
	#pragma region

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static ColumnWiseMatrix<ms, md> Multiply(const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	* Same version as above, but gives the possibility of reusing the output buffer
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static void Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static Vector<ms, md> Dot(const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	* Same version as above, but gives the possibility of reusing the output buffer
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static Vector<ms, md> Dot(Vector<ms, md>& out, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);

	/**
	* Solve A * X = B, B is overwritten
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static void Solve(ColumnWiseMatrix<ms, md>& rhs, const CompressedSparseRowMatrix<ms, md>& lhs, LinearSystemSolverType solver = LinearSystemSolverType::Lu);

	/**
	* Solve A * x = b, b is overwritten
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Solve(Vector<ms, md>& rhs, const CompressedSparseRowMatrix<ms, md>& lhs, LinearSystemSolverType solver = LinearSystemSolverType::Lu);

#pragma endregion

	#pragma region Type aliases

	using GpuIntegerSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Int>;
	using GpuSingleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Float>;
	using GpuFloatSparseMatrix = GpuSingleSparseMatrix;
	using GpuDoubleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Device, MathDomain::Double>;

	using CudaCpuIntegerSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Int>;
	using CudaCpuSingleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Float>;
	using CudaCpuFloatSparseMatrix = CudaCpuSingleSparseMatrix;
	using CudaCpuDoubleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Host, MathDomain::Double>;

	using TestIntegerSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Test, MathDomain::Int>;
	using TestSingleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Test, MathDomain::Float>;
	using TestFloatSparseMatrix =TestSingleSparseMatrix;
	using TestDoubleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Test, MathDomain::Double>;

	using MklIntegerSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Mkl, MathDomain::Int>;
	using MklSingleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Mkl, MathDomain::Float>;
	using MklFloatSparseMatrix = MklSingleSparseMatrix;
	using MklDoubleSparseMatrix = CompressedSparseRowMatrix<MemorySpace::Mkl, MathDomain::Double>;

	namespace gpu
	{
		using smat = cl::GpuFloatSparseMatrix;
		using dsmat = cl::GpuDoubleSparseMatrix ;
		using ismat = cl::GpuIntegerSparseMatrix;
	}

	// by default we're gonna be using GPU
	using smat = gpu::smat;
	using dsmat = gpu::dsmat;
	using ismat = gpu::ismat;

	namespace cudaCpu
	{
		using smat = cl::CudaCpuSingleSparseMatrix;
		using dsmat = cl::CudaCpuDoubleSparseMatrix;
		using ismat = cl::CudaCpuIntegerSparseMatrix;
	}

	namespace mkl
	{
		using smat = cl::MklSingleSparseMatrix;
		using dsmat = cl::MklDoubleSparseMatrix;
		using ismat = cl::MklIntegerSparseMatrix;
	}

	namespace test
	{
		using smat = cl::TestSingleSparseMatrix;
		using dsmat = cl::TestDoubleSparseMatrix;
		using ismat = cl::TestIntegerSparseMatrix;
	}

	#pragma endregion
}

// give possibility of avoiding writing cl::
namespace mkl
{
	using smat = cl::MklSingleSparseMatrix;
	using dsmat = cl::MklDoubleSparseMatrix;
	using ismat = cl::MklIntegerSparseMatrix;
}

namespace test
{
	using smat = cl::TestSingleSparseMatrix;
	using dsmat = cl::TestDoubleSparseMatrix;
	using ismat = cl::TestIntegerSparseMatrix;
}

#include <CompressedSparseRowMatrix.tpp>
