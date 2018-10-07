#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <DeviceManager.h>
#include <DeviceManagerHelper.h>
#include <Exception.h>
#include <Types.h>

namespace cl
{
	// This is required for using types defined in children classes (which are incomplete types)
	#pragma region Type mapping from MathDomain to C++ type

	template <MathDomain mathDomain>
	struct Traits {
		using stdType = void;
	};

	template <>
	struct Traits<MathDomain::Double> {
		using stdType = double;
	};

	template <>
	struct Traits<MathDomain::Float> {
		using stdType = float;
	};

	template <>
	struct Traits<MathDomain::Int> {
		using stdType = int;
	};

	#pragma endregion 

	#pragma region Type mapping from C++ type to MathDomain

	template <typename T>
	struct _Traits {
		static constexpr MathDomain clType = MathDomain::Null;
	};

	template <>
	struct _Traits<double> {
		static constexpr MathDomain clType = MathDomain::Double;
	};

	template <>
	struct _Traits<float> {
		static constexpr MathDomain clType = MathDomain::Float;
	};

	template <>
	struct _Traits<int> {
		static constexpr MathDomain clType = MathDomain::Int;
	};

	#pragma endregion 

	/*
	 * CRTP implementation
	 */
	template<typename BufferImpl, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class IBuffer
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;

		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(const IBuffer<BufferImpl, ms, md>& rhs) = delete;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		IBuffer& operator=(IBuffer<BufferImpl, ms, md>&& rhs) = delete;

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		void ReadFrom(const IBuffer<bi, ms, md>& rhs);
		template<typename T>
		void ReadFrom(const std::vector<T>& rhs);

		void Set(const stdType value) const;

		void LinSpace(const stdType x0, const stdType x1) const;

		void RandomUniform(const unsigned seed = 1234) const;

		void RandomGaussian(const unsigned seed = 1234) const;

		virtual std::vector<stdType> Get() const;

		virtual void Print(const std::string& label = "") const = 0;

		virtual std::ostream& ToOutputStream(std::ostream& os) const = 0;
		virtual void ToBinaryFile(const std::string& fileName, const std::string mode = "w") const = 0;

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		friend std::ostream& operator<<(std::ostream& os, const IBuffer<bi, ms, md>& buffer);

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator==(const IBuffer<bi, ms, md>& rhs) const;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator!=(const IBuffer<bi, ms, md>& rhs) const { return !(*this == rhs); }

		unsigned size() const noexcept { return GetBuffer().size; };

		virtual ~IBuffer();

		#pragma region Linear Algebra

		IBuffer& operator +=(const IBuffer& rhs);
		IBuffer& operator -=(const IBuffer& rhs);
		IBuffer& operator %=(const IBuffer& rhs);  // element-wise product

		IBuffer& AddEqual(const IBuffer& rhs, const double alpha = 1.0);
		IBuffer& Scale(const double alpha);

		int AbsoluteMinimumIndex() const;
		int AbsoluteMaximumIndex() const;
		stdType MinimumInAbsoluteValue() const;
		stdType MaximumInAbsoluteValue() const;

		#pragma endregion
		
		protected:
		virtual const MemoryBuffer& GetBuffer() const noexcept = 0;
		explicit IBuffer(const bool isOwner);

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
		void dtor(MemoryBuffer buffer);

		static void Alloc(MemoryBuffer& buffer);

		const bool isOwner;
	};

	namespace detail
	{

		template<MathDomain md>
		void Fill(std::vector<typename Traits<md>::stdType>& dest, const MemoryBuffer& source)
		{
			using stdType = typename Traits<md>::stdType;
			const stdType* const ptr = reinterpret_cast<const stdType* const>(source.pointer);
			for (size_t i = 0; i < dest.size(); i++)
				dest[i] = ptr[i];
		}

		template<MathDomain md>
		void Fill(std::vector<typename Traits<md>::stdType>& dest, const MemoryTile& source)
		{
			using stdType = typename Traits<md>::stdType;
			const stdType* const ptr = reinterpret_cast<const stdType* const>(source.pointer);
			for (size_t j = 0; j < source.nCols; j++)
				for (size_t i = 0; i < source.nRows; i++)
					dest[i + source.nRows * j] = ptr[i + source.nRows * j];
		}
	}

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static void Scale(IBuffer<BufferImpl, ms, md>& lhs, const double alpha);

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const IBuffer<BufferImpl, ms, md>& vec, const std::string& label = "");

	template<typename T>
	static void Print(const std::vector<T>& vec, const std::string& label = "");

	template<typename T>
	static void Print(const std::vector<T>& mat, const unsigned nRows, const unsigned nCols, const std::string& label = "");

    #pragma region Serialization

	template<typename T>
	static std::ostream& VectorToOutputStream(const std::vector<T>& vec, std::ostream& os);

	template<typename T>
	static std::istream& VectorFromInputStream(std::vector<T>& vec, std::istream& is);

	template<typename T>
	static std::ostream& MatrixToOutputStream(const std::vector<T>& vec, const unsigned nRows, const unsigned nCols, std::ostream& os);

	template<typename T>
	static std::istream& MatrixFromInputStream(std::vector<T>& vec, unsigned& nRows, unsigned& nCols, std::istream& is);

	template<typename T>
	static void VectorToBinaryFile(const std::vector<T>& vec, const std::string& fileName, const std::string mode = "w");

	template<typename T>
	static void VectorFromBinaryFile(std::vector<T>& vec, const std::string& fileName, const bool useMemoryMapping = false);

	template<typename T>
	static void MatrixToBinaryFile(const std::vector<T>& mat, const unsigned nRows, const unsigned nCols, const std::string& fileName, const std::string mode = "w");

	template<typename T>
	static void MatrixFromBinaryFile(std::vector<T>& mat, unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool useMemoryMapping = false);

    #pragma endregion
}

#include <IBuffer.tpp>

