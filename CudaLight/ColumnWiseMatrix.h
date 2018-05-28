#pragma once

#include <vector>
#include <string>
#include <memory>

#include <IBuffer.h>
#include <Types.h>
#include <Vector.h>

namespace cl
{
	template<MemorySpace memorySpace, MathDomain mathDomain>
	class CompressedSparseRowMatrix;

	template<MemorySpace memorySpace, MathDomain mathDomain>
	class ColumnWiseMatrix : public IBuffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		using stdType = typename Traits<mathDomain>::stdType;
		friend class IBuffer<ColumnWiseMatrix<memorySpace, mathDomain>, memorySpace, mathDomain>;
		template<MemorySpace ms, MathDomain md>
		friend class CompressedSparseRowMatrix;
		 
		ColumnWiseMatrix(const unsigned nRows, const unsigned nCols);

		ColumnWiseMatrix(const unsigned nRows, const unsigned nCols, const stdType value);

		ColumnWiseMatrix(const unsigned nRows);

		ColumnWiseMatrix(const ColumnWiseMatrix& rhs);

		ColumnWiseMatrix(const std::vector<stdType>& rhs, const unsigned nRows, const unsigned nCols);

		ColumnWiseMatrix(const Vector<memorySpace, mathDomain>& rhs);		

		using IBuffer<ColumnWiseMatrix, memorySpace, mathDomain>::ReadFrom;
		void ReadFrom(const Vector<memorySpace, mathDomain>& rhs);

		using IBuffer<ColumnWiseMatrix, memorySpace, mathDomain>::Get;
		std::vector<typename Traits<mathDomain>::stdType> Get(const unsigned column) const;

		void MakeIdentity();

		void Set(const Vector<memorySpace, mathDomain>& columnVector, const unsigned column);

		void Print(const std::string& label = "") const override;

		virtual ~ColumnWiseMatrix() = default;

		unsigned nRows() const noexcept { return buffer.nRows; }
		unsigned nCols() const noexcept { return buffer.nCols; }

		std::vector<std::shared_ptr<Vector<memorySpace, mathDomain>>> columns;

		#pragma region Linear Algebra

		ColumnWiseMatrix operator +(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator -(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator %(const ColumnWiseMatrix& rhs) const;

		ColumnWiseMatrix operator *(const ColumnWiseMatrix& rhs) const;
		ColumnWiseMatrix operator *=(const ColumnWiseMatrix& rhs) const;

		Vector<memorySpace, mathDomain> operator *(const Vector<memorySpace, mathDomain>& rhs) const;

		ColumnWiseMatrix Multiply(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		/**
		 * Same version as above, but gives the possibility of reusing the output buffer
		 */
		void Multiply(ColumnWiseMatrix& out, const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		
		Vector<memorySpace, mathDomain> Dot(const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;
		/**
		* Same version as above, but gives the possibility of reusing the output buffer
		*/
		void Dot(Vector<memorySpace, mathDomain>& out, const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0) const;

		ColumnWiseMatrix Add(const ColumnWiseMatrix& rhs, const double alpha = 1.0) const;

		/**
		* Invert inplace - WARNING, use Solve for higher performance
		*/
		void Invert(const MatrixOperation lhsOperation = MatrixOperation::None);

		/**
		* Solve A * X = B, B is overwritten
		*/
		void Solve(const ColumnWiseMatrix& rhs, const MatrixOperation lhsOperation = MatrixOperation::None);

		/**
		* Solve A * x = b, b is overwritten
		*/
		void Solve(const Vector<memorySpace, mathDomain>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None);

		#pragma endregion

		#pragma region Enable shared ptr contruction

	private:
		struct EnableSharedPtr {};
	public:
		explicit ColumnWiseMatrix(EnableSharedPtr, const MemoryTile& buffer)
			: ColumnWiseMatrix(buffer)
		{

		}
		static std::shared_ptr<ColumnWiseMatrix> make_shared(const MemoryTile& buffer) {
			return std::make_shared<ColumnWiseMatrix>(EnableSharedPtr(), buffer);
		}

		#pragma endregion

		const MemoryBuffer& GetBuffer() const noexcept override { return buffer; }
	protected:
		explicit ColumnWiseMatrix(const MemoryTile& buffer);
		

		MemoryTile buffer;
	};

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Copy(const ColumnWiseMatrix<ms, md>& source);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Eye(const unsigned nRows);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> LinSpace(const typename Traits<md>::stdType x0, const typename Traits<md>::stdType x1, const unsigned nRows, const unsigned nCols);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> RandomUniform(const unsigned nRows, const unsigned nCols, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> RandomGaussian(const unsigned nRows, const unsigned nCols, const unsigned seed = 1234);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Print(const ColumnWiseMatrix<ms, md>& vec, const std::string& label = "");

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Add(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	ColumnWiseMatrix<ms, md> Multiply(const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	 * Same version as above, but gives the possibility of reusing the output buffer
	 */
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Multiply(ColumnWiseMatrix<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const ColumnWiseMatrix<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const MatrixOperation rhsOperation = MatrixOperation::None, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	Vector<ms, md> Dot(const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);
	/**
	* Same version as above, but gives the possibility of reusing the output buffer
	*/
	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Dot(Vector<ms, md>& out, const ColumnWiseMatrix<ms, md>& lhs, const Vector<ms, md>& rhs, const MatrixOperation lhsOperation = MatrixOperation::None, const double alpha = 1.0);

	template<MemorySpace ms = MemorySpace::Device, MathDomain md = MathDomain::Float>
	void Scale(ColumnWiseMatrix<ms, md>& lhs, const double alpha);

	#pragma region Type aliases

	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Int> GpuIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float> GpuSingleMatrix;
	typedef GpuSingleMatrix GpuFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double> GpuDoubleMatrix;

	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Int> CpuIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Float> CpuSingleMatrix;
	typedef CpuSingleVector CpuFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Double> CpuDoubleMatrix;

	typedef GpuSingleMatrix mat;
	typedef GpuDoubleMatrix dmat;
	typedef GpuIntegerMatrix imat;

	#pragma endregion
}

#include <ColumnWiseMatrix.tpp>


