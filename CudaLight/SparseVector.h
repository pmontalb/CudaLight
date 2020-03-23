#pragma once

#include <string>

#include <Buffer.h>
#include <Types.h>
#include <Vector.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class SparseVector : public Buffer<SparseVector<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class Buffer<SparseVector, memorySpace, mathDomain>;

		SparseVector(const unsigned size, const Vector<memorySpace, MathDomain::Int>& nonZeroIndices);
		SparseVector(const unsigned size, const Vector<memorySpace, MathDomain::Int>& nonZeroIndices, const stdType value);
		// copy denseVector to host, numerically finds the non-zero indices, and then copy back to device
		explicit SparseVector(const Vector<memorySpace, mathDomain>& denseVector);

		SparseVector(const SparseVector& rhs);
		SparseVector(SparseVector&& rhs) noexcept;

        inline ~SparseVector() override
        {
            this-> dtor(_buffer);
            _buffer.pointer = 0;
        }

		const MemoryBuffer& GetBuffer() const noexcept final { return values.GetBuffer(); }
		MemoryBuffer& GetBuffer() noexcept final { return values.GetBuffer(); }

		std::vector<stdType> Get() const final;
		void Print(const std::string& label = "") const final;
		std::ostream& ToOutputStream(std::ostream&) const final { throw std::logic_error("Not Implemented"); }
		void ToBinaryFile(const std::string&, const bool, const std::string) const final	{ throw std::logic_error("Not Implemented"); }

		#pragma region Dense-Sparse Linear Algebra

		Vector<memorySpace, mathDomain> operator +(const Vector<memorySpace, mathDomain>& rhs) const;
		Vector<memorySpace, mathDomain> operator -(const Vector<memorySpace, mathDomain>& rhs) const;
		Vector<memorySpace, mathDomain> Add(const Vector<memorySpace, mathDomain>& rhs, const double alpha = 1.0) const;

		#pragma endregion

		#pragma region Linear Algebra

		/**
		 * WARNING: this assumes the same sparsity pattern between operands
		 * NB: buffer.pointer is the same as values.pointer, so it doesn't require any additional care
		 */
		SparseVector operator +(const SparseVector& rhs) const;
		SparseVector operator -(const SparseVector& rhs) const;
		SparseVector operator %(const SparseVector& rhs) const;  // element-wise product
		SparseVector Add(const SparseVector& rhs, const double alpha = 1.0) const;

		#pragma endregion
	protected:
		SparseVector(const unsigned size, const unsigned nNonZeros);
		SparseVector(const unsigned size, const unsigned nNonZeros, const stdType value);
        using Buffer<SparseVector, memorySpace, mathDomain>::Alloc;


    public:
        unsigned denseSize;  // used only when converting to dense
    protected:
        SparseMemoryBuffer _buffer {};
        Vector<memorySpace, mathDomain> values {};
        Vector<memorySpace, MathDomain::Int> nonZeroIndices {};

	private:
		/**
		 * buffer.pointer <- values.pointer
		 * buffer.indices <- nonZeroIndices.pointer
		 */
		void SyncPointers();
	};

	#pragma region Type aliases

	using GpuIntegerSparseVector = SparseVector<MemorySpace::Device, MathDomain::Int>;
	using GpuSingleSparseVector = SparseVector<MemorySpace::Device, MathDomain::Float>;
	using GpuFloatSparseVector = GpuSingleSparseVector;
	using GpuDoubleSparseVector = SparseVector<MemorySpace::Device, MathDomain::Double>;

	using CudaCpuIntegerSparseVector = SparseVector<MemorySpace::Host, MathDomain::Int>;
	using CudaCpuSingleSparseVector = SparseVector<MemorySpace::Host, MathDomain::Float>;
	using CudaCpuFloatSparseVector = CudaCpuSingleSparseVector;
	using CudaCpuDoubleSparseVector = SparseVector<MemorySpace::Host, MathDomain::Double>;

	using TestIntegerSparseVector = SparseVector<MemorySpace::Test, MathDomain::Int>;
	using TestSingleSparseVector = SparseVector<MemorySpace::Test, MathDomain::Float>;
	using TestFloatSparseVector =TestSingleSparseVector;
	using TestDoubleSparseVector = SparseVector<MemorySpace::Test, MathDomain::Double>;

	using MklIntegerSparseVector = SparseVector<MemorySpace::Mkl, MathDomain::Int>;
	using MklSingleSparseVector = SparseVector<MemorySpace::Mkl, MathDomain::Float>;
	using MklFloatSparseVector = MklSingleSparseVector;
	using MklDoubleSparseVector = SparseVector<MemorySpace::Mkl, MathDomain::Double>;

	namespace gpu
	{
		using svec = cl::GpuSingleSparseVector;
		using dsvec = cl::GpuDoubleSparseVector;
		using isvec = cl::GpuIntegerSparseVector;
	}

	// by default we're gonna be using GPU
	using svec = gpu::svec;
	using dsvec = gpu::dsvec;
	using isvec = gpu::isvec;

	namespace cudaCpu
	{
		using svec = cl::CudaCpuSingleSparseVector;
		using sdvec = cl::CudaCpuDoubleSparseVector;
		using isvec = cl::CudaCpuIntegerSparseVector;
	}

	namespace mkl
	{
		using svec = cl::MklSingleSparseVector;
		using dsvec = cl::MklDoubleSparseVector;
		using isvec = cl::MklIntegerSparseVector;
	}

	namespace test
	{
		using svec = cl::TestSingleSparseVector;
		using dsvec = cl::TestDoubleSparseVector;
		using isvec = cl::TestIntegerSparseVector;
	}

	#pragma endregion
}

// give possibility of avoiding writing cl::
namespace mkl
{
	using svec = cl::mkl::svec;
	using dsvec = cl::mkl::dsvec;
	using isvec = cl::mkl::isvec;
}

namespace test
{
	using svec = cl::test::svec;
	using dsvec = cl::test::dsvec;
	using isvec = cl::test::isvec;
}

#include <SparseVector.tpp>
