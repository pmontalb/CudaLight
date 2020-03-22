#pragma once

#include <vector>
#include <string>
#include <memory>

#include <HostBuffer.h>
#include <Types.h>
#include <HostVector.h>

namespace cl { namespace host
{
	template<MemorySpace memorySpace, MathDomain mathDomain>
	class CompressedSparseRowMatrix;

	template<MemorySpace memorySpace, MathDomain mathDomain>
	class ColumnWiseMatrix : public Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		static_assert(memorySpace != MemorySpace::Host && memorySpace != MemorySpace::Device, "Unsupported memory space");

		using stdType = typename Traits<mathDomain>::stdType;
		friend class Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>;
		template<MemorySpace ms, MathDomain md>
		friend class CompressedSparseRowMatrix;
		template<MemorySpace ms, MathDomain md>
		friend class Tensor;
		 
		ColumnWiseMatrix(const unsigned nRows, const unsigned nCols);

		ColumnWiseMatrix(const unsigned nRows, const unsigned nCols, const stdType value);

		ColumnWiseMatrix(const unsigned nRows);

		ColumnWiseMatrix(const ColumnWiseMatrix& rhs);
		ColumnWiseMatrix(ColumnWiseMatrix&& rhs) noexcept;
		
		// initialise a sub matrix
		ColumnWiseMatrix(const ColumnWiseMatrix& rhs, const size_t colStart, const size_t colEnd);
		// reshape a vector into a matrix
		ColumnWiseMatrix(const Vector<memorySpace, mathDomain>& rhs, const size_t startOffset, const size_t nRows, const size_t nCols);

		ColumnWiseMatrix(const std::vector<stdType>& rhs, const unsigned nRows, const unsigned nCols);
		ColumnWiseMatrix(const std::string& fileName, bool useMemoryMapping);

		ColumnWiseMatrix(const Vector<memorySpace, mathDomain>& rhs);
		
		using Buffer<ColumnWiseMatrix, memorySpace, mathDomain>::ReadFrom;
		void ReadFrom(const Vector<memorySpace, mathDomain>& rhs);

		using Buffer<ColumnWiseMatrix, memorySpace, mathDomain>::Get;
		std::vector<stdType> Get(const unsigned column) const;

		void MakeIdentity();

		Vector<memorySpace, mathDomain> Flatten() const;
		
		void RandomShuffleColumns(const unsigned seed);

		using Buffer<ColumnWiseMatrix, memorySpace, mathDomain>::Set;
		void Set(const Vector<memorySpace, mathDomain>& columnVector, const unsigned column);

		void Print(const std::string& label = "") const final;
		std::ostream& ToOutputStream(std::ostream& os) const final;
		void ToBinaryFile(const std::string& fileName, const bool compressed = false, const std::string mode = "w") const final;

		template<MemorySpace ms, MathDomain md>
		friend std::ostream& operator<<(std::ostream& os, const ColumnWiseMatrix& buffer);

        inline ~ColumnWiseMatrix() override
        {
            this->dtor(_buffer);
            _buffer.pointer = 0;
        }

		unsigned nRows() const noexcept { return _buffer.nRows; }
		unsigned nCols() const noexcept { return _buffer.nCols; }

		std::vector<std::shared_ptr<Vector<memorySpace, mathDomain>>> columns {};

		#pragma region Linear Algebra
		
		using Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>::LinSpace;
		using Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomUniform;
		using Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomGaussian;
		using Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>::Scale;
		
		void ScaleColumns(const Vector<memorySpace, mathDomain>& alpha);
		
		ColumnWiseMatrix operator +(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator -(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator %(const ColumnWiseMatrix& rhs) const;

		ColumnWiseMatrix operator *(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator *=(const ColumnWiseMatrix& rhs);

		Vector<memorySpace, mathDomain> operator *(const Vector<memorySpace, mathDomain>& rhs) const;
		
		ColumnWiseMatrix& AddEqualMatrix(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 1.0);
		
		ColumnWiseMatrix& AddEqualBroadcast(const Vector<memorySpace, mathDomain>& rhs, const bool rowWise = true, const double alpha = 1.0);
		ColumnWiseMatrix& AddEqualBroadcast(const Vector<memorySpace, mathDomain>& rhs, const Vector<memorySpace, mathDomain>& cache, const bool rowWise = true, const double alpha = 1.0);
		
		/**
		* A = alpha * B * C + beta * A
		*/
		ColumnWiseMatrix Multiply(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;
		/**
		 * Same version as above, but gives the possibility of reusing the output buffer
		 */
		void Multiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;
		
		/**
		* A = alpha * B * C + beta * A
		*/
		ColumnWiseMatrix SubMultiply(const ColumnWiseMatrix& rhs, const size_t rowStart, const size_t colStart, const size_t nRows, const size_t nCols, const size_t colRhsStart, const size_t nColsRhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;
		/**
		 * Same version as above, but gives the possibility of reusing the output buffer
		 */
		void SubMultiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const size_t rowStart, const size_t colStart, const size_t nRows, const size_t nCols, const size_t colRhsStart, const size_t nColsRhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;
		
		/**
		* y = alpha * A * x + beta * y
		*/
		Vector<memorySpace, mathDomain> Dot(const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		void Dot(Vector<memorySpace, mathDomain>& out, const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0) const;

		static ColumnWiseMatrix KroneckerProduct(const Vector<memorySpace, mathDomain>& lhs, const Vector<memorySpace, mathDomain>& rhs, const double alpha = 1.0);
		static void KroneckerProduct(ColumnWiseMatrix& out, const Vector<memorySpace, mathDomain>& lhs, const Vector<memorySpace, mathDomain>& rhs, const double alpha = 1.0);
		
		/**
		* y = alpha * A * x + beta * y
		*/
		Vector<memorySpace, mathDomain> RowWiseSum() const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		void RowWiseSum(Vector<memorySpace, mathDomain>& out) const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer and the cache needed to do the actual sum
		*/
		void RowWiseSum(Vector<memorySpace, mathDomain>& out, Vector<memorySpace, mathDomain>& cache) const;
		
		/**
* y = alpha * A * x + beta * y
*/
		Vector<memorySpace, mathDomain> ColumnWiseSum() const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		void ColumnWiseSum(Vector<memorySpace, mathDomain>& out) const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer and the cache needed to do the actual sum
		*/
		void ColumnWiseSum(Vector<memorySpace, mathDomain>& out, Vector<memorySpace, mathDomain>& cache) const;
		
		/*
		* A = alpha * B + beta * A
		*/
		ColumnWiseMatrix Add(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 1.0) const;

		/**
		* Invert inplace - WARNING, use Solve for higher performance
		*/
		void Invert(const MatrixOperation lhsOperation = MatrixOperation::None);

		/**
		* Solve A * X = B, B is overwritten
		*/
		void Solve(ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None) const;

		/**
		* Solve A * x = b, b is overwritten
		*/
		void Solve(Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None) const;

		Vector<memorySpace, MathDomain::Int> ColumnWiseArgAbsMinimum() const;
		void ColumnWiseArgAbsMinimum(Vector<memorySpace, MathDomain::Int>& out) const;

		Vector<memorySpace, MathDomain::Int> ColumnWiseArgAbsMaximum() const;
		void ColumnWiseArgAbsMaximum(Vector<memorySpace, MathDomain::Int>& out) const;

		#pragma endregion

		#pragma region Enable shared ptr contruction

	private:
		struct EnableSharedPtr {};
	public:
		explicit ColumnWiseMatrix(EnableSharedPtr, const MemoryTile& buffer)
			: ColumnWiseMatrix(buffer)
		{

		}
		static std::shared_ptr<ColumnWiseMatrix> make_shared(const MemoryTile& buffer) {
			return std::make_shared<ColumnWiseMatrix>(EnableSharedPtr(), buffer);
		}

		#pragma endregion
		
		MemoryBuffer& GetBuffer() noexcept final { return _buffer; }
		const MemoryBuffer& GetBuffer() const noexcept final { return _buffer; }
		MemoryTile& GetTile() noexcept { return _buffer; }
		const MemoryTile& GetTile() const noexcept { return _buffer; }
	protected:
		explicit ColumnWiseMatrix(const MemoryTile& buffer);
		
		MemoryTile _buffer = MemoryTile(0, 0, 0, memorySpace, mathDomain);
		
	private:
		void SetUp(const size_t nCols);
		
	public:
		// static functions
		static ColumnWiseMatrix Copy(const ColumnWiseMatrix& source);
		
		static ColumnWiseMatrix Eye(const unsigned nRows);
		
		static ColumnWiseMatrix LinSpace(const stdType x0, const stdType x1, const unsigned nRows, const unsigned nCols);
		
		static ColumnWiseMatrix RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned seed);
		
		static ColumnWiseMatrix RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned seed);
		
		static void RandomShuffleColumns(ColumnWiseMatrix& v, const unsigned seed = 1234);
		
		static void RandomShuffleColumnsPair(ColumnWiseMatrix& m1, ColumnWiseMatrix& m2, const unsigned seed = 1234);
		
		static void Print(const ColumnWiseMatrix& mat, const std::string& label = "");
		
		static std::ostream& MatrixToOutputStream(const ColumnWiseMatrix& mat, std::ostream& os);

		static void MatrixToBinaryFile(const ColumnWiseMatrix& vec, const std::string& fileName, const bool tranposed = true, const bool compressed = false, const std::string mode = "w");

		static ColumnWiseMatrix MatrixFromInputStream(std::istream& is);
		
		static ColumnWiseMatrix MatrixFromBinaryFile(const std::string& fileName, const bool compressed = false, const bool useMemoryMapping = false);
		
		static void MatrixFromBinaryFile(ColumnWiseMatrix& out, const std::string& fileName, const bool compressed = false, const std::string mode = "w");
		
		static ColumnWiseMatrix Add(const ColumnWiseMatrix& lhs, const ColumnWiseMatrix& rhs, const double alpha = 1.0);
		
		static ColumnWiseMatrix Multiply(const ColumnWiseMatrix& lhs, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0);
		/**
		 * Same version as above, but gives the possibility of reusing the output buffer
		 */
		static void Multiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& lhs, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0);
		
		static Vector<memorySpace, mathDomain> Dot(const ColumnWiseMatrix& lhs, const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		static void Dot(Vector<memorySpace, mathDomain>& out, const ColumnWiseMatrix& lhs, const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
		
		static void Scale(ColumnWiseMatrix& lhs, const double alpha);
		
		static Vector<memorySpace, MathDomain::Float> MakeTriple(const Vector<memorySpace, mathDomain>& x, const Vector<memorySpace, mathDomain>& y, const Vector<memorySpace, mathDomain>& z);
		
		static void MakeTriple(Vector<memorySpace, MathDomain::Float>& triple, const Vector<memorySpace, mathDomain>& x, const Vector<memorySpace, mathDomain>& y, const Vector<memorySpace, mathDomain>& z);
		
	};
	
	#pragma region Type aliases
		
	typedef ColumnWiseMatrix<MemorySpace::Test, MathDomain::Int> DebugIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Test, MathDomain::Float> DebugSingleMatrix;
	typedef DebugSingleMatrix DebugFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Test, MathDomain::Double> DebugDoubleMatrix;

	typedef ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Int> MklIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Float> MklSingleMatrix;
	typedef MklSingleMatrix MklFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Double> MklDoubleMatrix;

	#pragma endregion
}}

namespace cl { namespace mkl {
	typedef cl::host::MklSingleMatrix mat;
	typedef cl::host::MklDoubleMatrix dmat;
	typedef cl::host::MklIntegerMatrix imat;
}}

#include <HostColumnWiseMatrix.tpp>


