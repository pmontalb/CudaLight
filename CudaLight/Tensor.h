#pragma once

#include <vector>
#include <string>
#include <memory>

#include <Types.h>
#include <ColumnWiseMatrix.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class Tensor : public Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>;

		Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices);

		Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, stdType value);

		Tensor(const unsigned nRows, const unsigned nMatrices);

		Tensor(const Tensor& rhs);
		Tensor(Tensor&& rhs) noexcept;
		
		template<typename T>
		Tensor(const std::vector<T>& rhs, const unsigned nRows, const unsigned nCols, const unsigned nMatrices);
		
		explicit Tensor(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs);

		explicit Tensor(const Vector<memorySpace, mathDomain>& rhs);
		
		Tensor(const Vector<memorySpace, mathDomain>& rhs, const size_t startOffset, const size_t nRows, const size_t nCols, const size_t nMatrices) noexcept;

		using Buffer<Tensor, memorySpace, mathDomain>::ReadFrom;
		void ReadFrom(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs);
		void ReadFrom(const Vector<memorySpace, mathDomain>& rhs);

		using Buffer<Tensor, memorySpace, mathDomain>::Get;
		std::vector<typename Traits<mathDomain>::stdType> Get(const unsigned matrix) const;
		std::vector<typename Traits<mathDomain>::stdType> Get(const unsigned matrix, const unsigned column) const;

		using Buffer<Tensor, memorySpace, mathDomain>::Set;
		void Set(const ColumnWiseMatrix<memorySpace, mathDomain>& matrixBuffer, const unsigned matrix);
		void Set(const Vector<memorySpace, mathDomain>& columnVector, const unsigned column, const unsigned matrix);

		void Print(const std::string& label = "") const final;
		std::ostream& ToOutputStream(std::ostream&) const final { throw std::logic_error("Not Implemented"); }
		void ToBinaryFile(const std::string&, const bool, const std::string) const final { throw std::logic_error("Not Implemented"); }

        inline ~Tensor() override
        {
            this->dtor(_buffer);
            _buffer.pointer = 0;
        }

		unsigned nRows() const noexcept { return _buffer.nRows; }
		unsigned nCols() const noexcept { return _buffer.nCols; }
		unsigned nMatrices() const noexcept { return _buffer.nCubes; }

		std::vector<std::shared_ptr<ColumnWiseMatrix<memorySpace, mathDomain>>> matrices {};
	
		#pragma region Linear Algebra

		using Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>::LinSpace;
		using Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomUniform;
		using Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomGaussian;
		using Buffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>::Scale;

		Tensor operator +(const Tensor& rhs) const;
		Tensor operator -(const Tensor& rhs) const;
		Tensor operator %(const Tensor& rhs) const;

		Tensor Add(const Tensor& rhs, const double alpha = 1.0) const;
		
		ColumnWiseMatrix<memorySpace, mathDomain> CubeWiseSum() const;
		void CubeWiseSum(ColumnWiseMatrix<memorySpace, mathDomain>& out) const;
		void CubeWiseSum(ColumnWiseMatrix<memorySpace, mathDomain>& out, const CompressedSparseRowMatrix<memorySpace, mathDomain>& onesCache) const;
		
		ColumnWiseMatrix<memorySpace, mathDomain> MatrixSum() const;
		void MatrixSum(ColumnWiseMatrix<memorySpace, mathDomain>& out) const;
		void MatrixSum(ColumnWiseMatrix<memorySpace, mathDomain>& out, Vector<memorySpace, mathDomain>& cacheOnes) const;
		
		// NB: this computes KroneckerProduct(lhs->columns[i], rhs->columns[i]) and stores the result in this->matrices[i], so we're transposing the cubes!
		static Tensor KroneckerProduct(const ColumnWiseMatrix<memorySpace, mathDomain>& lhs, const ColumnWiseMatrix<memorySpace, mathDomain>& rhs, const double alpha = 1.0);
		static void KroneckerProduct(Tensor& out, const ColumnWiseMatrix<memorySpace, mathDomain>& lhs, const ColumnWiseMatrix<memorySpace, mathDomain>& rhs, const double alpha = 1.0);
		static void AccumulateKroneckerProduct(ColumnWiseMatrix<memorySpace, mathDomain>& out, const ColumnWiseMatrix<memorySpace, mathDomain>& lhs, const ColumnWiseMatrix<memorySpace, mathDomain>& rhs, const double alpha = 1.0);
		
		#pragma endregion

		static Tensor Copy(const Tensor& source);

		static Tensor LinSpace(const stdType x0, const stdType x1, const unsigned nRows, const unsigned nCols, const unsigned nMatrices);

		static Tensor RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed);

		static Tensor RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed);

		static void Print(const Tensor& vect, const std::string& label = "");

		static Tensor Add(const Tensor& lhs, const Tensor& rhs, const double alpha = 1.0);

		static void Scale(Tensor& lhs, const double alpha);

		MemoryBuffer& GetBuffer() noexcept final { return _buffer; }
		const MemoryBuffer& GetBuffer() const noexcept final { return _buffer; }
		const MemoryCube& GetCube() const noexcept { return _buffer; }
		MemoryCube& GetCube() noexcept { return _buffer; }
	private:
		void SetUp(const size_t nMatrices);
	protected:
		explicit Tensor(const MemoryCube& buffer);

		MemoryCube _buffer;
	};

	#pragma region Type aliases

	using GpuIntegerTensor = Tensor<MemorySpace::Device, MathDomain::Int>;
	using GpuSingleTensor = Tensor<MemorySpace::Device, MathDomain::Float>;
	using GpuFloatTensor = GpuSingleTensor;
	using GpuDoubleTensor = Tensor<MemorySpace::Device, MathDomain::Double>;

	using CudaCpuIntegerTensor = Tensor<MemorySpace::Host, MathDomain::Int>;
	using CudaCpuSingleTensor = Tensor<MemorySpace::Host, MathDomain::Float>;
	using CudaCpuFloatTensor = CudaCpuSingleTensor;
	using CudaCpuDoubleTensor = Tensor<MemorySpace::Host, MathDomain::Double>;

	using TestIntegerTensor = Tensor<MemorySpace::Test, MathDomain::Int>;
	using TestSingleTensor = Tensor<MemorySpace::Test, MathDomain::Float>;
	using TestFloatTensor =TestSingleTensor;
	using TestDoubleTensor = Tensor<MemorySpace::Test, MathDomain::Double>;

	using MklIntegerTensor = Tensor<MemorySpace::Mkl, MathDomain::Int>;
	using MklSingleTensor = Tensor<MemorySpace::Mkl, MathDomain::Float>;
	using MklFloatTensor = MklSingleTensor;
	using MklDoubleTensor = Tensor<MemorySpace::Mkl, MathDomain::Double>;

	using OpenBlasIntegerTensor = Tensor<MemorySpace::OpenBlas, MathDomain::Int>;
	using OpenBlasSingleTensor = Tensor<MemorySpace::OpenBlas, MathDomain::Float>;
	using OpenBlasFloatTensor = OpenBlasSingleTensor;
	using OpenBlasDoubleTensor = Tensor<MemorySpace::OpenBlas, MathDomain::Double>;

	using GenericBlasIntegerTensor = Tensor<MemorySpace::GenericBlas, MathDomain::Int>;
	using GenericBlasSingleTensor = Tensor<MemorySpace::GenericBlas, MathDomain::Float>;
	using GenericBlasFloatTensor = GenericBlasSingleTensor;
	using GenericBlasDoubleTensor = Tensor<MemorySpace::GenericBlas, MathDomain::Double>;

	namespace gpu
	{
		using ten = cl::GpuSingleTensor;
		using dten = cl::GpuDoubleTensor;
		using iten = cl::GpuIntegerTensor;
	}

	// by default we're gonna be using GPU
	using ten = gpu::ten;
	using dten = gpu::dten;
	using iten = gpu::iten;

	namespace cudaCpu
	{
		using ten = cl::CudaCpuSingleTensor;
		using dten = cl::CudaCpuDoubleTensor;
		using iten = cl::CudaCpuIntegerTensor;
	}

	namespace mkl
	{
		using ten = cl::MklSingleTensor;
		using dten = cl::MklDoubleTensor;
		using iten = cl::MklIntegerTensor;
	}

	namespace oblas
	{
		using ten = cl::OpenBlasSingleTensor;
		using dten = cl::OpenBlasDoubleTensor;
		using iten = cl::OpenBlasIntegerTensor;
	}

	namespace gblas
	{
		using ten = cl::GenericBlasSingleTensor;
		using dten = cl::GenericBlasDoubleTensor;
		using iten = cl::GenericBlasIntegerTensor;
	}
	
	namespace test
	{
		using ten = cl::TestSingleTensor;
		using dten = cl::TestDoubleTensor;
		using iten = cl::TestIntegerTensor;
	}

	#pragma endregion
}

// give possibility of avoiding writing cl::
namespace mkl
{
	using ten = cl::mkl::ten;
	using dten = cl::mkl::dten;
	using iten = cl::mkl::iten;
}

namespace oblas
{
	using ten = cl::oblas::ten;
	using dten = cl::oblas::dten;
	using iten = cl::oblas::iten;
}

namespace gblas
{
	using ten = cl::gblas::ten;
	using dten = cl::gblas::dten;
	using iten = cl::gblas::iten;
}

namespace test
{
	using ten = cl::test::ten;
	using dten = cl::test::dten;
	using iten = cl::test::iten;
}

#include <Tensor.tpp>
