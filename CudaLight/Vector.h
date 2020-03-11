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
		Vector(const Vector& rhs, const size_t start, const size_t end) noexcept;
		Vector(Vector&& rhs) noexcept;
		explicit Vector(const std::vector<stdType>& rhs);
        explicit Vector(const std::string& fileName, bool useMemoryMapping = false);

		using IBuffer<Vector, memorySpace, mathDomain>::Set;

		inline ~Vector() override
		{
            this->dtor(_buffer);
            _buffer.pointer = 0;
		}
		
		void RandomShuffle(const unsigned seed = 1234);
		
		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(const Vector& rhs) = delete;
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(Vector&& rhs) = delete;

		void Print(const std::string& label = "") const override final;
		std::ostream& ToOutputStream(std::ostream& os) const override final;
		virtual void ToBinaryFile(const std::string& fileName, const bool compressed = false, const std::string mode = "w") const override final;

		template<MemorySpace ms, MathDomain md>
		friend std::ostream& operator<<(std::ostream& os, const Vector& buffer);

		#pragma region Linear Algebra
		
		using IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>::LinSpace;
		using IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomUniform;
		using IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>::RandomGaussian;
		using IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>::Scale;

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
		
		MemoryBuffer& GetBuffer() noexcept override final { return _buffer; }
		const MemoryBuffer& GetBuffer() const noexcept override final { return _buffer; }
	protected:
		
		using IBuffer<Vector, memorySpace, mathDomain>::IBuffer;

		Vector() : IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>(true) {}
		explicit Vector(const MemoryBuffer& buffer);
	private:
		MemoryBuffer _buffer {};
		
	public:
		// static functions
		static Vector Copy(const Vector& source);
		
		static Vector LinSpace(const typename Traits<mathDomain>::stdType x0, const typename Traits<mathDomain>::stdType x1, const unsigned size);
		
		static Vector RandomUniform(const unsigned size, const unsigned seed);
		
		static Vector RandomGaussian(const unsigned size, const unsigned seed);
		
		static void RandomShuffle(Vector& v, const unsigned seed = 1234);
		
		static void RandomShufflePair(Vector& v1, Vector& v2, const unsigned seed = 1234);
		
		static void Print(const Vector& vec, const std::string& label = "");
		
		static std::ostream& VectorToOutputStream(const Vector& vec, std::ostream& os);
		
		static void VectorToBinaryFile(const Vector& vec, const std::string& fileName, const bool compressed = false, const std::string mode = "w");
		
		static Vector VectorFromInputStream(std::istream& is);
		
		static Vector VectorFromBinaryFile(const std::string& fileName, const bool compressed = false, const bool useMemoryMapping = false);
		
		static Vector Add(const Vector& lhs, const Vector& rhs, const double alpha = 1.0);
		
		static void Scale(Vector& lhs, const double alpha);
		
		static Vector<memorySpace, MathDomain::Float> MakePair(const Vector& x, const Vector& y);
		
		static void MakePair(Vector<memorySpace, MathDomain::Float>& pair, const Vector& x, const Vector& y);
	};

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

