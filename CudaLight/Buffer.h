#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <DeviceManager.h>
#include <DeviceManagerHelper.h>
#include <Exception.h>
#include <IBuffer.h>
#include <Types.h>

namespace cl
{
	/*
	 * CRTP implementation
	 */
	template<typename BufferImpl, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class Buffer: public IBuffer<memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;

		~Buffer() override = default;

		// For avoiding unnecessary checks and overheads, it's not possible to use operator=
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Buffer& operator=(const Buffer<BufferImpl, ms, md>& rhs) = delete;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		Buffer& operator=(Buffer<BufferImpl, ms, md>&& rhs) = delete;

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		void ReadFrom(const Buffer<bi, ms, md>& rhs);
		template<typename T>
		void ReadFrom(const std::vector<T>& rhs);

		void Set(const stdType value) final;

		void Reciprocal() final;

		void LinSpace(const stdType x0, const stdType x1) final;

		void RandomUniform(const unsigned seed = 1234) final;

		void RandomGaussian(const unsigned seed = 1234) final;

		virtual std::vector<stdType> Get() const override;

		template<typename bi, MemorySpace ms, MathDomain md>
		friend std::ostream& operator<<(std::ostream& os, const Buffer<bi, ms, md>& buffer);

		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator==(const Buffer<bi, ms, md>& rhs) const;
		template<typename bi, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
		bool operator!=(const Buffer<bi, ms, md>& rhs) const
		{
			return !(*this == rhs);
		}

		unsigned size() const noexcept final { return this->GetBuffer().size; }

#pragma region Linear Algebra

		IBuffer<memorySpace, mathDomain>& operator+=(const IBuffer<memorySpace, mathDomain>& rhs) final;
		IBuffer<memorySpace, mathDomain>& operator-=(const IBuffer<memorySpace, mathDomain>& rhs) final;
		IBuffer<memorySpace, mathDomain>& operator%=(const IBuffer<memorySpace, mathDomain>& rhs) final;	// element-wise product

		IBuffer<memorySpace, mathDomain>& AddEqual(const IBuffer<memorySpace, mathDomain>& rhs, const double alpha = 1.0) final;
		IBuffer<memorySpace, mathDomain>& Scale(const double alpha) final;
		IBuffer<memorySpace, mathDomain>& ElementWiseProduct(const IBuffer<memorySpace, mathDomain>& rhs, const double alpha = 1.0) final;

		int AbsoluteMinimumIndex() const final;
		int AbsoluteMaximumIndex() const final;
		stdType AbsoluteMinimum() const final;
		stdType AbsoluteMaximum() const final;
		stdType Minimum() const final;
		stdType Maximum() const final;
		stdType Sum() const final;
		stdType EuclideanNorm() const final;
		int CountEquals(const IBuffer<memorySpace, mathDomain>& rhs) const final;
		int CountEquals(const IBuffer<memorySpace, mathDomain>& rhs, MemoryBuffer& cacheCount, MemoryBuffer& cacheSum, MemoryBuffer& oneElementCache) const final;

#pragma endregion

		inline bool OwnsMemory() const noexcept { return _isOwner; }

	protected:
		explicit Buffer(const bool isOwner);
		explicit Buffer(Buffer&& buffer) noexcept;

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
		void dtor(MemoryBuffer& buffer);

		static void Alloc(MemoryBuffer& buffer);

		bool _isOwner;
	};

	namespace detail
	{
		template<MathDomain md>
		void Fill(std::vector<typename Traits<md>::stdType>& dest, const MemoryBuffer& source)
		{
			using stdType = typename Traits<md>::stdType;
			const auto* ptr = reinterpret_cast<const stdType*>(source.pointer);
			for (size_t i = 0; i < dest.size(); i++)
				dest[i] = ptr[i];
		}

		template<MathDomain md>
		void Fill(std::vector<typename Traits<md>::stdType>& dest, const MemoryTile& source)
		{
			using stdType = typename Traits<md>::stdType;
			const auto* ptr = reinterpret_cast<const stdType*>(source.pointer);
			for (size_t j = 0; j < source.nCols; j++)
				for (size_t i = 0; i < source.nRows; i++)
					dest[i + source.nRows * j] = ptr[i + source.nRows * j];
		}
	}	 // namespace detail

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	static void Scale(Buffer<BufferImpl, ms, md>& lhs, const double alpha);

	template<typename BufferImpl, MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const Buffer<BufferImpl, ms, md>& v, const std::string& label = "");

	template<typename T>
	static void Print(const std::vector<T>& v, const std::string& label = "");

	template<typename T>
	static void Print(const std::vector<T>& m, const unsigned nRows, const unsigned nCols, const std::string& label = "");

#pragma region Serialization

	template<typename T>
	static std::ostream& VectorToOutputStream(const std::vector<T>& v, std::ostream& os);

	template<typename T>
	static std::istream& VectorFromInputStream(std::vector<T>& v, std::istream& is);

	template<typename T>
	static std::ostream& MatrixToOutputStream(const std::vector<T>& m, const unsigned nRows, const unsigned nCols, std::ostream& os);

	template<typename T>
	static std::istream& MatrixFromInputStream(std::vector<T>& m, unsigned& nRows, unsigned& nCols, std::istream& is);

	template<typename T>
	static void VectorToBinaryFile(const std::vector<T>& v, const std::string& fileName, const bool compressed = false, const std::string mode = "w");

	template<typename T>
	static void VectorFromBinaryFile(std::vector<T>& v, const std::string& fileName, const bool compressed = false, const bool useMemoryMapping = false);

	template<typename T>
	static std::vector<T> VectorFromBinaryFile(const std::string& fileName, const bool compressed = false, const bool useMemoryMapping = false);

	template<typename T>
	static void MatrixToBinaryFile(const std::vector<T>& m, unsigned nRows, unsigned nCols, const std::string& fileName, const bool tranpose = true, const bool compressed = false, const std::string mode = "w");

	template<typename T>
	static void MatrixFromBinaryFile(std::vector<T>& m, unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool transpose = true, const bool compressed = false, const bool useMemoryMapping = false);

	template<typename T>
	static std::vector<T> MatrixFromBinaryFile(unsigned& nRows, unsigned& nCols, const std::string& fileName, const bool compressed = false, const bool useMemoryMapping = false);

#pragma endregion
}	 // namespace cl

#include <Buffer.tpp>
