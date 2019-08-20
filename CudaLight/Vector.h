#pragma once

#include <string>
#include <memory>

#include <IBuffer.h>
#include <Types.h>

namespace cl
{
	template<MemorySpace memorySpace, MathDomain mathDomain>
	class SparseVector;
	template<MemorySpace memorySpace, MathDomain mathDomain>
	class CompressedSparseRowMatrix;

	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class Vector : public IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;

		friend class IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>;
		template<MemorySpace ms, MathDomain md>
		friend class SparseVector;
		template<MemorySpace ms, MathDomain md>
		friend class CompressedSparseRowMatrix;

		explicit Vector(const unsigned size);

		Vector(const unsigned size, const stdType value);
		Vector(const Vector& rhs);
		explicit Vector(const std::vector<stdType>& rhs);
        explicit Vector(const std::string& fileName, bool useMemoryMapping = false);

		using IBuffer<Vector, memorySpace, mathDomain>::Set;

		virtual ~Vector() = default;
		
		void RandomShuffle(const unsigned seed = 1234) const;
		
		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(const Vector<ms, md>& rhs) = delete;
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(Vector<ms, md>&& rhs) = delete;

		void Print(const std::string& label = "") const override final;
		std::ostream& ToOutputStream(std::ostream& os) const override final;
		virtual void ToBinaryFile(const std::string& fileName, const std::string mode = "w") const override final;

		template<MemorySpace ms, MathDomain md>
		friend std::ostream& operator<<(std::ostream& os, const Vector<ms, md>& buffer);

		#pragma region Linear Algebra

		Vector operator +(const Vector& rhs) const;
		Vector operator -(const Vector& rhs) const;
		Vector operator %(const Vector& rhs) const;  // element-wise product

		Vector Add(const Vector& rhs, const double alpha = 1.0) const;

		#pragma endregion

		#pragma region Enable shared ptr contruction

		private:
			struct EnableSharedPtr {};
		public:
			explicit Vector(EnableSharedPtr, const MemoryBuffer& buffer)
				: Vector(buffer)
			{
			
			}
			static std::shared_ptr<Vector> make_shared(const MemoryBuffer& buffer) {
				return std::make_shared<Vector>(EnableSharedPtr(), buffer);
			}

		#pragma endregion

		const MemoryBuffer& GetBuffer() const noexcept override final { return buffer; }
	protected:
		using IBuffer<Vector, memorySpace, mathDomain>::IBuffer;

		Vector() : IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>(true) {};
		explicit Vector(const MemoryBuffer& buffer);
	private:

		MemoryBuffer buffer;
	};

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Copy(const Vector<ms, md>& source);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned size);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> RandomUniform(const unsigned size, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> RandomGaussian(const unsigned size, const unsigned seed = 1234);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void RandomShuffle(Vector<ms, md>& v, const unsigned seed = 1234);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void RandomShufflePair(Vector<ms, md>& v1, Vector<ms, md>& v2, const unsigned seed = 1234);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const Vector<ms, md>& vec, const std::string& label = "");

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	std::ostream& VectorToOutputStream(const Vector<ms, md>& vec, std::ostream& os);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void VectorToBinaryFile(const Vector<ms, md>& vec, const std::string& fileName, const std::string mode = "w");

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> VectorFromInputStream(std::istream& is);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> VectorFromBinaryFile(const std::string& fileName, const bool useMemoryMapping = false);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Add(const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Scale(Vector<ms, md>& lhs, const double alpha);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, MathDomain::Float> MakePair(const Vector<ms, md>& x, const Vector<ms, md>& y);
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void MakePair(Vector<ms, MathDomain::Float>& pair, const Vector<ms, md>& x, const Vector<ms, md>& y);

	#pragma region Type aliases

	typedef Vector<MemorySpace::Device, MathDomain::Int> GpuIntegerVector;
	typedef Vector<MemorySpace::Device, MathDomain::Float> GpuSingleVector;
	typedef GpuSingleVector GpuFloatVector;
	typedef Vector<MemorySpace::Device, MathDomain::Double> GpuDoubleVector;

	typedef Vector<MemorySpace::Host, MathDomain::Int> CpuIntegerVector;
	typedef Vector<MemorySpace::Host, MathDomain::Float> CpuSingleVector;
	typedef CpuSingleVector CpuFloatVector;
	typedef Vector<MemorySpace::Host, MathDomain::Double> CpuDoubleVector;
	
	typedef GpuSingleVector vec;
	typedef GpuDoubleVector dvec;
	typedef GpuIntegerVector ivec;

	#pragma endregion
}

#include <Vector.tpp>

