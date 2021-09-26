#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <Traits.h>
#include <Types.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class IBuffer
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;

		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(const IBuffer<ms, md>& rhs) = delete;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(IBuffer<ms, md>&& rhs) = delete;

		virtual void Set(const stdType value) = 0;

		virtual void Reciprocal() = 0;

		virtual void LinSpace(const stdType x0, const stdType x1) = 0;

		virtual void RandomUniform(const unsigned seed = 1234) = 0;

		virtual void RandomGaussian(const unsigned seed = 1234) = 0;

		virtual std::vector<stdType> Get() const = 0;

		virtual void Print(const std::string& label = "") const = 0;

		virtual std::ostream& ToOutputStream(std::ostream& os) const = 0;
		virtual void ToBinaryFile(const std::string& fileName, const bool compressed = false, const std::string mode = "w") const = 0;

		virtual unsigned size() const noexcept = 0;

		virtual ~IBuffer() = default;

		virtual IBuffer& operator+=(const IBuffer& rhs) = 0;
		virtual IBuffer& operator-=(const IBuffer& rhs) = 0;
		virtual IBuffer& operator%=(const IBuffer& rhs) = 0;	// element-wise product

		virtual IBuffer& AddEqual(const IBuffer& rhs, const double alpha = 1.0) = 0;
		virtual IBuffer& Scale(const double alpha) = 0;
		virtual IBuffer& ElementWiseProduct(const IBuffer& rhs, const double alpha = 1.0) = 0;

		virtual int AbsoluteMinimumIndex() const = 0;
		virtual int AbsoluteMaximumIndex() const = 0;
		virtual stdType AbsoluteMinimum() const = 0;
		virtual stdType AbsoluteMaximum() const = 0;
		virtual stdType Minimum() const = 0;
		virtual stdType Maximum() const = 0;
		virtual stdType Sum() const = 0;
		virtual stdType EuclideanNorm() const = 0;
		virtual int CountEquals(const IBuffer& rhs) const = 0;
		virtual int CountEquals(const IBuffer& rhs, MemoryBuffer& cacheCount, MemoryBuffer& cacheSum, MemoryBuffer& oneElementCache) const = 0;

		virtual MemoryBuffer& GetBuffer() noexcept = 0;
		virtual const MemoryBuffer& GetBuffer() const noexcept = 0;
	};
}	 // namespace cl
