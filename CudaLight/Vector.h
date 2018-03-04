#pragma once

#include <vector>
#include <string>

#include <DeviceManager.h>
#include <IBuffer.h>
#include <Exception.h>
#include <Types.h>
#include <type_traits>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class Vector : public IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class IBuffer<Vector<memorySpace, mathDomain>, memorySpace, mathDomain>;

		explicit Vector(const unsigned size);

		Vector(const unsigned size, const double value);

		Vector(const Vector& rhs);

		virtual ~Vector() = default;

		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(const Vector<ms, md>& rhs) = delete;
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Vector& operator=(Vector<ms, md>&& rhs) = delete;

		std::vector<double> Get() const override;

		void Print(const std::string& label = "") const override;

		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator==(const Vector<ms, md>& rhs) const;
		template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator!=(const Vector<ms, md>& rhs) const { return !(*this == rhs); }

		#pragma region Linear Algebra

		Vector operator +(const Vector& rhs) const;
		Vector operator -(const Vector& rhs) const;
		Vector operator %(const Vector& rhs) const;  // element-wise product
		Vector Add(const Vector& rhs, const double alpha = 1.0) const;

		#pragma endregion

	protected:
		using IBuffer<Vector, memorySpace, mathDomain>::IBuffer;

		const MemoryBuffer& GetBuffer() const noexcept override { return buffer; }

		MemoryBuffer buffer = MemoryBuffer();
	};

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Copy(const Vector<ms, md>& source);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> LinSpace(const double x0, const double x1, const unsigned size = 100);
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> RandomUniform(const unsigned size = 100, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> RandomGaussian(const unsigned size = 100, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const Vector<ms, md>& vec, const std::string& label = "");
	
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Add(const Vector<ms, md>& lhs, const Vector<ms, md>& rhs, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Scale(Vector<ms, md>& lhs, const double alpha);

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

