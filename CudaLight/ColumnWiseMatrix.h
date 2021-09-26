#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Buffer.h>
#include <Types.h>
#include <Vector.h>

namespace cl
{
	template<MemorySpace memorySpace, MathDomain mathDomain>
	class CompressedSparseRowMatrix;

	template<MemorySpace memorySpace, MathDomain mathDomain>
	class ColumnWiseMatrix: public Buffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
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

		ColumnWiseMatrix operator+(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator-(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator%(const ColumnWiseMatrix& rhs) const;

		ColumnWiseMatrix operator*(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator*=(const ColumnWiseMatrix& rhs);

		Vector<memorySpace, mathDomain> operator*(const Vector<memorySpace, mathDomain>& rhs) const;

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
		void Solve(ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, LinearSystemSolverType solver = LinearSystemSolverType::Lu) const;

		/**
		 * Solve A * x = b, b is overwritten
		 */
		void Solve(Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, LinearSystemSolverType solver = LinearSystemSolverType::Lu) const;

		Vector<memorySpace, MathDomain::Int> ColumnWiseArgAbsMinimum() const;
		void ColumnWiseArgAbsMinimum(Vector<memorySpace, MathDomain::Int>& out) const;

		Vector<memorySpace, MathDomain::Int> ColumnWiseArgAbsMaximum() const;
		void ColumnWiseArgAbsMaximum(Vector<memorySpace, MathDomain::Int>& out) const;

#pragma endregion

#pragma region Enable shared ptr contruction

	private:
		struct EnableSharedPtr
		{
		};

	public:
		explicit ColumnWiseMatrix(EnableSharedPtr, const MemoryTile& buffer) : ColumnWiseMatrix(buffer) {}
		static std::shared_ptr<ColumnWiseMatrix> make_shared(const MemoryTile& buffer) { return std::make_shared<ColumnWiseMatrix>(EnableSharedPtr(), buffer); }

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

		static void MatrixToBinaryFile(const ColumnWiseMatrix& vect, const std::string& fileName, const bool tranposed = true, const bool compressed = false, const std::string mode = "w");

		static ColumnWiseMatrix MatrixFromInputStream(std::istream& is);

		static ColumnWiseMatrix MatrixFromBinaryFile(const std::string& fileName, const bool transposed = false, const bool compressed = false, const bool useMemoryMapping = false);

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

	using GpuIntegerMatrix = ColumnWiseMatrix<MemorySpace::Device, MathDomain::Int>;
	using GpuSingleMatrix = ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float>;
	using GpuFloatMatrix = GpuSingleMatrix;
	using GpuDoubleMatrix = ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double>;

	using CudaCpuIntegerMatrix = ColumnWiseMatrix<MemorySpace::Host, MathDomain::Int>;
	using CudaCpuSingleMatrix = ColumnWiseMatrix<MemorySpace::Host, MathDomain::Float>;
	using CudaCpuFloatMatrix = CudaCpuSingleMatrix;
	using CudaCpuDoubleMatrix = ColumnWiseMatrix<MemorySpace::Host, MathDomain::Double>;

	using TestIntegerMatrix = ColumnWiseMatrix<MemorySpace::Test, MathDomain::Int>;
	using TestSingleMatrix = ColumnWiseMatrix<MemorySpace::Test, MathDomain::Float>;
	using TestFloatMatrix = TestSingleMatrix;
	using TestDoubleMatrix = ColumnWiseMatrix<MemorySpace::Test, MathDomain::Double>;

	using MklIntegerMatrix = ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Int>;
	using MklSingleMatrix = ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Float>;
	using MklFloatMatrix = MklSingleMatrix;
	using MklDoubleMatrix = ColumnWiseMatrix<MemorySpace::Mkl, MathDomain::Double>;

	using OpenBlasIntegerMatrix = ColumnWiseMatrix<MemorySpace::OpenBlas, MathDomain::Int>;
	using OpenBlasSingleMatrix = ColumnWiseMatrix<MemorySpace::OpenBlas, MathDomain::Float>;
	using OpenBlasFloatMatrix = OpenBlasSingleMatrix;
	using OpenBlasDoubleMatrix = ColumnWiseMatrix<MemorySpace::OpenBlas, MathDomain::Double>;

	using GenericBlasIntegerMatrix = ColumnWiseMatrix<MemorySpace::GenericBlas, MathDomain::Int>;
	using GenericBlasSingleMatrix = ColumnWiseMatrix<MemorySpace::GenericBlas, MathDomain::Float>;
	using GenericBlasFloatMatrix = GenericBlasSingleMatrix;
	using GenericBlasDoubleMatrix = ColumnWiseMatrix<MemorySpace::GenericBlas, MathDomain::Double>;

	namespace gpu
	{
		using mat = cl::GpuSingleMatrix;
		using dmat = cl::GpuDoubleMatrix;
		using imat = cl::GpuIntegerMatrix;
	}	 // namespace gpu

	// by default we're gonna be using GPU
	using mat = gpu::mat;
	using dmat = gpu::dmat;
	using imat = gpu::imat;

	namespace cudaCpu
	{
		using mat = cl::CudaCpuSingleMatrix;
		using dmat = cl::CudaCpuDoubleMatrix;
		using imat = cl::CudaCpuIntegerMatrix;
	}	 // namespace cudaCpu

	namespace mkl
	{
		using mat = cl::MklSingleMatrix;
		using dmat = cl::MklDoubleMatrix;
		using imat = cl::MklIntegerMatrix;
	}	 // namespace mkl

	namespace oblas
	{
		using mat = cl::OpenBlasSingleMatrix;
		using dmat = cl::OpenBlasDoubleMatrix;
		using imat = cl::OpenBlasIntegerMatrix;
	}	 // namespace oblas

	namespace gblas
	{
		using mat = cl::GenericBlasSingleMatrix;
		using dmat = cl::GenericBlasDoubleMatrix;
		using imat = cl::GenericBlasIntegerMatrix;
	}	 // namespace gblas

	namespace test
	{
		using mat = cl::TestSingleMatrix;
		using dmat = cl::TestDoubleMatrix;
		using imat = cl::TestIntegerMatrix;
	}	 // namespace test

#pragma endregion
}	 // namespace cl

// give possibility of avoiding writing cl::
namespace mkl
{
	using mat = cl::mkl::mat;
	using dmat = cl::mkl::dmat;
	using imat = cl::mkl::imat;
}	 // namespace mkl

namespace oblas
{
	using mat = cl::oblas::mat;
	using dmat = cl::oblas::dmat;
	using imat = cl::oblas::imat;
}	 // namespace oblas

namespace gblas
{
	using mat = cl::gblas::mat;
	using dmat = cl::gblas::dmat;
	using imat = cl::gblas::imat;
}	 // namespace gblas

namespace test
{
	using mat = cl::test::mat;
	using dmat = cl::test::dmat;
	using imat = cl::test::imat;
}	 // namespace test

#include <ColumnWiseMatrix.tpp>
