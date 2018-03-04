#pragma once

#include <vector>
#include <string>

#include <DeviceManager.h>
#include <Exception.h>
#include <Types.h>

namespace cl
{
	/*
	 * CRTP implementation
	 */
	template<typename BufferImpl, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class IBuffer
	{
	public:
		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(const IBuffer<BufferImpl, ms, md>& rhs) = delete;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(IBuffer<BufferImpl, ms, md>&& rhs) = delete;

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		void ReadFrom(const IBuffer<bi, ms, md>& rhs);

		template<typename T>
		void ReadFrom(const std::vector<T>& rhs)
		{
			if (!buffer.pointer)
				throw std::exception("Buffer needs to be initialised first!");

			MemoryBuffer rhsBuf;
			if (std::is_same<double, T>::value)
				rhsBuf = MemoryBuffer(static_cast<ptr_t>(rhs.data()), rhs.size(), MemorySpace::Host, MathDomain::Double);
			else if (std::is_same<float, T>::value)
				rhsBuf = MemoryBuffer(static_cast<ptr_t>(rhs.data()), rhs.size(), MemorySpace::Host, MathDomain::Float);
			else if (std::is_same<int, T>::value)
				rhsBuf = MemoryBuffer(static_cast<ptr_t>(rhs.data()), rhs.size(), MemorySpace::Host, MathDomain::Int);
			else
				throw NotSupportedException();
			dm::detail::AutoCopy(buffer, rhsBuf);
		}

		void Set(const double value) const
		{
			dm::detail::Initialize(buffer, value);
		}

		void LinSpace(const double x0, const double x1) const;

		void RandomUniform(const unsigned seed = 1234) const;

		void RandomGaussian(const unsigned seed = 1234) const;

		virtual std::vector<double> Get() const = 0;

		virtual void Print(const std::string& label = "") const = 0;

		virtual unsigned size() const noexcept { return GetBuffer().size; };

		virtual ~IBuffer();

		#pragma region Linear Algebra

		IBuffer& operator +=(const IBuffer& rhs);
		IBuffer& operator -=(const IBuffer& rhs);
		IBuffer& operator %=(const IBuffer& rhs);  // element-wise product

		IBuffer& AddEqual(const IBuffer& rhs, const double alpha = 1.0);
		IBuffer& Scale(const double alpha);

		#pragma endregion

	protected:
		explicit IBuffer(const bool isOwner = true);
		virtual const MemoryBuffer& GetBuffer() const noexcept = 0;

		static constexpr double GetTolerance()
		{
			switch (mathDomain)
			{
			case MathDomain::Int:
				return 0;
			case MathDomain::Float:
				return 1e-7;
			case MathDomain::Double:
				return 1e-15;
			default:
				return 0;
			}
		}

		void ctor(MemoryBuffer& buffer);
	private:
		const bool isOwner = true;
	};

	namespace detail
	{
		template<typename T>
		void FillImpl(std::vector<double>& dest, const MemoryBuffer& source)
		{
			const T* const ptr = reinterpret_cast<const T* const>(source.pointer);
			for (size_t i = 0; i < dest.size(); i++)
				dest[i] = ptr[i];
		}

		void Fill(std::vector<double>& dest, const MemoryBuffer& source)
		{
			switch (source.mathDomain)
			{
			case MathDomain::Int:
				FillImpl<int>(dest, source);
				break;
			case MathDomain::Float:
				FillImpl<float>(dest, source);
				break;
			case MathDomain::Double:
				FillImpl<double>(dest, source);
				break;
			default:
				throw NotSupportedException();
			}
		}
	}

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const IBuffer<BufferImpl, ms, md>& vec, const std::string& label = "");

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Scale(IBuffer<BufferImpl, ms, md>& lhs, const double alpha);
}

#include <IBuffer.tpp>

