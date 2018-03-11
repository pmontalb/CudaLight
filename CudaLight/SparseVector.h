#pragma once

#include <string>
#include <memory>

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
		SparseVector(const Vector<memorySpace, mathDomain>& denseVector);

		SparseVector(const SparseVector& rhs);

		virtual ~SparseVector() = default;

		const MemoryBuffer& GetBuffer() const noexcept override { return values.GetBuffer(); }

		std::vector<typename Traits<mathDomain>::stdType> Get() const override;
		void Print(const std::string& label = "") const override;
	protected:
		SparseVector(const unsigned size, const unsigned nNonZeros);
		SparseVector(const unsigned size, const unsigned nNonZeros, const stdType value);

		SparseMemoryBuffer buffer;

		Vector<memorySpace, mathDomain> values;
		Vector<memorySpace, MathDomain::Int> nonZeroIndices;
		unsigned denseSize;  // used only when converting to dense

		using IBuffer<SparseVector, memorySpace, mathDomain>::Alloc;
	};

	#pragma region Type aliases

	typedef SparseVector<MemorySpace::Device, MathDomain::Int> GpuIntegerSparseVector;
	typedef SparseVector<MemorySpace::Device, MathDomain::Float> GpuSingleSparseVector;
	typedef GpuSingleVector GpuFloatSparseVector;
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

 namespace cl
{
	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices)
		 : IBuffer(false),  // SparseVector doesn't allocate its memory in its buffer!
		   values(nonZeroIndices.size()), nonZeroIndices(nonZeroIndices), 
		   denseSize(size), 
		   buffer(0, nonZeroIndices.size(), nonZeroIndices.GetBuffer().pointer, ms, md)
	 {
	 }

	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const unsigned size, const Vector<ms, MathDomain::Int>& nonZeroIndices, const typename Traits<md>::stdType value)
		 : SparseVector(size, nonZeroIndices)
	 {
		 dm::detail::Initialize(values.GetBuffer(), value);
	 }

	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const unsigned size, const unsigned nNonZeros)
		 : IBuffer(false), // SparseVector doesn't allocate its memory in its buffer!
		   values(nNonZeros), nonZeroIndices(nNonZeros), denseSize(size)
	 {
		 assert(size != 0);
		 assert(nNonZeros != 0);
		 assert(nNonZeros < size);
	 }

	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const unsigned size, const unsigned nNonZeros, const typename Traits<md>::stdType value)
		 : SparseVector(size, nNonZeros)
	 {
		 dm::detail::Initialize(static_cast<MemoryBuffer>(buffer), value);
	 }

	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const Vector<ms, md>& denseVector)
		 : IBuffer(false), denseSize(denseVector.size())
	 {
		 const auto hostDenseVector = denseVector.Get();

		 std::vector<int> nonZeroIndices;
		 std::vector<typename Traits<md>::stdType> nonZeroValues;
		 for (int i = 0; i < hostDenseVector.size(); ++i)
		 {
			 if (hostDenseVector[i] > 1e-7)
			 {
				 nonZeroIndices.push_back(i);
				 nonZeroValues.push_back(hostDenseVector[i]);
			 }
		 }

		 values.buffer = MemoryBuffer(0, nonZeroValues.size(), ms, md);
		 Alloc(values.buffer);
		 values.ReadFrom(nonZeroValues);
		 
		 this->nonZeroIndices.buffer = MemoryBuffer(0, nonZeroIndices.size(), ms, md);
		 Alloc(this->nonZeroIndices.buffer);
		 this->nonZeroIndices.ReadFrom(nonZeroIndices);
	 }

	 template< MemorySpace ms, MathDomain md>
	 SparseVector<ms, md>::SparseVector(const SparseVector& rhs)
		 : IBuffer(false), // SparseVector doesn't allocate its memory in its buffer!
	       values(rhs.values), nonZeroIndices(rhs.nonZeroIndices), denseSize(rhs.denseSize)
	 {
		 
	 }

	 template< MemorySpace ms, MathDomain md>
	 std::vector<typename Traits<md>::stdType> SparseVector<ms, md>::Get() const
	 {
		 auto values = this->values.Get();
		 auto indices = this->nonZeroIndices.Get();

		 std::vector<typename Traits<md>::stdType> ret(denseSize, 0.0);
		 for (int i = 0; i < indices.size(); i++)
			 ret[indices[i]] = values[i];

		 return ret;
	 }

	 template< MemorySpace ms, MathDomain md>
	 void SparseVector<ms, md>::Print(const std::string& label) const
	 {
		 auto vec = Get();
		 cl::Print(vec);
	 }
}