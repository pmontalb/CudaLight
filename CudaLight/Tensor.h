#pragma once

#include <vector>
#include <string>
#include <memory>

#include <Types.h>
#include <ColumnWiseMatrix.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class Tensor : public IBuffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class IBuffer<Tensor<memorySpace, mathDomain>, memorySpace, mathDomain>;

		Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices);

		Tensor(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, stdType value);

		Tensor(const unsigned nRows, const unsigned nMatrices);

		Tensor(const Tensor& rhs);
		template<typename T>
		Tensor(const std::vector<T>& rhs, const unsigned nRows, const unsigned nCols, const unsigned nMatrices);
		
		explicit Tensor(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs);

		explicit Tensor(const Vector<memorySpace, mathDomain>& rhs);

		using IBuffer<Tensor, memorySpace, mathDomain>::ReadFrom;
		void ReadFrom(const ColumnWiseMatrix<memorySpace, mathDomain>& rhs);
		void ReadFrom(const Vector<memorySpace, mathDomain>& rhs);

		using IBuffer<Tensor, memorySpace, mathDomain>::Get;
		std::vector<typename Traits<mathDomain>::stdType> Get(const unsigned matrix) const;
		std::vector<typename Traits<mathDomain>::stdType> Get(const unsigned matrix, const unsigned column) const;

		using IBuffer<Tensor, memorySpace, mathDomain>::Set;
		void Set(const ColumnWiseMatrix<memorySpace, mathDomain>& matrixBuffer, const unsigned matrix);
		void Set(const Vector<memorySpace, mathDomain>& columnVector, const unsigned column, const unsigned matrix);

		void Print(const std::string& label = "") const override final;
		std::ostream& ToOutputStream(std::ostream&) const override final { throw std::logic_error("Not Implemented"); };
		void ToBinaryFile(const std::string&, const std::string) const override final { throw std::logic_error("Not Implemented"); };

		virtual ~Tensor() = default;

		unsigned nRows() const noexcept { return buffer.nRows; }
		unsigned nCols() const noexcept { return buffer.nCols; }
		unsigned nMatrices() const noexcept { return buffer.nCubes; }

		std::vector<std::shared_ptr<ColumnWiseMatrix<memorySpace, mathDomain>>> matrices;
	
		#pragma region Linear Algebra

		Tensor operator +(const Tensor& rhs) const;
		Tensor operator -(const Tensor& rhs) const;
		Tensor operator %(const Tensor& rhs) const;

		Tensor Add(const Tensor& rhs, const double alpha = 1.0) const;

		#pragma endregion

		const MemoryBuffer& GetBuffer() const noexcept override final { return buffer; }
		const MemoryCube& GetCube() const noexcept { return buffer; }
	protected:

		explicit Tensor(const MemoryCube& buffer);

		MemoryCube buffer;

	};

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> Copy(const Tensor<ms, md>& source);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned nRows, const unsigned nCols, const unsigned nMatrices);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned nMatrices, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const Tensor<ms, md>& vec, const std::string& label = "");

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Tensor<ms, md> Add(const Tensor<ms, md>& lhs, const Tensor<ms, md>& rhs, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Scale(Tensor<ms, md>& lhs, const double alpha);

	#pragma region Type aliases

	typedef Tensor<MemorySpace::Device, MathDomain::Int> GpuIntegerTensor;
	typedef Tensor<MemorySpace::Device, MathDomain::Float> GpuSingleTensor;
	typedef GpuSingleTensor GpuFloatTensor;
	typedef Tensor<MemorySpace::Device, MathDomain::Double> GpuDoubleTensor;

	typedef Tensor<MemorySpace::Host, MathDomain::Int> CpuIntegerTensor;
	typedef Tensor<MemorySpace::Host, MathDomain::Float> CpuSingleTensor;
	typedef CpuSingleTensor CpuFloatTensor;
	typedef Tensor<MemorySpace::Host, MathDomain::Double> CpuDoubleTensor;

	typedef GpuSingleTensor ten;
	typedef GpuDoubleTensor dten;
	typedef GpuIntegerTensor iten;

	#pragma endregion
}

#include <Tensor.tpp>
