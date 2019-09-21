#pragma once

#include <string>

#include <IBuffer.h>
#include <Types.h>
#include <Vector.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class SparseVector : public IBuffer<SparseVector<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class IBuffer<SparseVector, memorySpace, mathDomain>;

		SparseVector(const unsigned size, const Vector<memorySpace, MathDomain::Int>& nonZeroIndices);
		SparseVector(const unsigned size, const Vector<memorySpace, MathDomain::Int>& nonZeroIndices, const stdType value);
		// copy denseVector to host, numerically finds the non-zero indices, and then copy back to device
		explicit SparseVector(const Vector<memorySpace, mathDomain>& denseVector);

		SparseVector(const SparseVector& rhs);

		virtual ~SparseVector() = default;

		const MemoryBuffer& GetBuffer() const noexcept override final { return values.GetBuffer(); }
		MemoryBuffer& GetBuffer() noexcept override final { return values.GetBuffer(); }

		std::vector<typename Traits<mathDomain>::stdType> Get() const override final;
		void Print(const std::string& label = "") const override final;
		std::ostream& ToOutputStream(std::ostream& os) const override final { throw std::logic_error("Not Implemented"); };
		void ToBinaryFile(const std::string& fileName, const bool compressed, const std::string mode) const override final	{ throw std::logic_error("Not Implemented"); };

		unsigned denseSize;  // used only when converting to dense

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

		SparseMemoryBuffer _buffer;

		Vector<memorySpace, mathDomain> values;
		Vector<memorySpace, MathDomain::Int> nonZeroIndices;

		using IBuffer<SparseVector, memorySpace, mathDomain>::Alloc;

	private:
		/**
		 * buffer.pointer <- values.pointer
		 * buffer.indices <- nonZeroIndices.pointer
		 */
		void SyncPointers();
	};

	#pragma region Type aliases

	typedef SparseVector<MemorySpace::Device, MathDomain::Int> GpuIntegerSparseVector;
	typedef SparseVector<MemorySpace::Device, MathDomain::Float> GpuSingleSparseVector;
	typedef GpuSingleSparseVector GpuFloatSparseVector;
	typedef SparseVector<MemorySpace::Device, MathDomain::Double> GpuDoubleSparseVector;

	typedef SparseVector<MemorySpace::Host, MathDomain::Int> CpuIntegerSparseVector;
	typedef SparseVector<MemorySpace::Host, MathDomain::Float> CpuSingleSparseVector;
	typedef CpuSingleSparseVector CpuFloatSparseVector;
	typedef SparseVector<MemorySpace::Host, MathDomain::Double> CpuDoubleSparseVector;

	typedef GpuSingleSparseVector svec;
	typedef GpuDoubleSparseVector dsvec;
	typedef GpuIntegerSparseVector isvec;

	#pragma endregion
}

#include <SparseVector.tpp>