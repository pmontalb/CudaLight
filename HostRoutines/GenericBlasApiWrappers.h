
#ifndef GENERIC_API_NAMESPACE
	#error "Wrong usage of this header"
#endif

#ifndef GENERIC_API_ROUTINES_NAMESPACE
	#error "Wrong usage of this header"
#endif

#define ROUTINES_NAMESPACE namespace GENERIC_API_ROUTINES_NAMESPACE
#define BLAS_NAMESPACE namespace GENERIC_API_NAMESPACE

#ifndef GENERIC_API_DEFINE

	namespace cl { namespace routines { ROUTINES_NAMESPACE
	{
		template<MathDomain md>
		static void Add(MemoryBuffer&, const MemoryBuffer&, const MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void AddEqual(MemoryBuffer&, const MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void AddEqualMatrix(MemoryTile&, const MemoryTile&, const MatrixOperation, const MatrixOperation, const double, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void Scale(MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ScaleColumns(MemoryTile&, const MemoryBuffer&)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ElementwiseProduct(MemoryBuffer&, const MemoryBuffer&, const MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void SubMultiply(MemoryTile&, const MemoryTile&, const MemoryTile&, const unsigned, const unsigned, const unsigned, const MatrixOperation, const MatrixOperation, const double, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void BatchedMultiply(MemoryCube&, const MemoryCube&, const MemoryCube&, const unsigned, const unsigned, const MatrixOperation, const MatrixOperation, const double, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void Dot(MemoryBuffer&, const MemoryTile&, const MemoryBuffer&, const MatrixOperation, const double, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void KroneckerProduct(MemoryTile&, const MemoryBuffer&, const MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void Solve(const MemoryTile&, MemoryTile&, const MatrixOperation, const LinearSystemSolverType)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ArgAbsMin(int&, const MemoryBuffer&)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ColumnWiseArgAbsMin(MemoryBuffer&, const MemoryTile&)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ArgAbsMax(int&, const MemoryBuffer&)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void ColumnWiseArgAbsMax(MemoryBuffer&, const MemoryTile&)
		{
			throw NotImplementedException();
		}

		// norm = ||x||_2
		template<MathDomain md>
		static void EuclideanNorm(double&, const MemoryBuffer&)
		{
			throw NotImplementedException();
		}
	}}}

#else

	#include <cmath>
	#include <cblas.h>
	#include <lapacke.h>

namespace cl { namespace routines { ROUTINES_NAMESPACE {
	static constexpr CBLAS_ORDER columnMajorLayout = { CBLAS_ORDER::CblasColMajor };
	static constexpr std::array<CBLAS_TRANSPOSE, 2> GENERIC_API_NAMESPACEOperationsEnum = { CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans };

	template<MathDomain md>
	static void Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha);

	template<>
	inline void Add<MathDomain::Float>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		Copy<MathDomain::Float>(z, y);
		cblas_saxpy(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), 1, reinterpret_cast<float*>(z.pointer), 1);
	}
	template<>
	inline void Add<MathDomain::Double>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		Copy<MathDomain::Double>(z, y);
		cblas_daxpy(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(x.pointer), 1, reinterpret_cast<double*>(z.pointer), 1);
	}

	template<MathDomain md>
	static void AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha);

	template<>
	inline void AddEqual<MathDomain::Float>(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
	{
		cblas_saxpy(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), 1, reinterpret_cast<float*>(z.pointer), 1);
	}
	template<>
	inline void AddEqual<MathDomain::Double>(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
	{
		cblas_daxpy(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(x.pointer), 1, reinterpret_cast<double*>(z.pointer), 1);
	}

	template<MathDomain md>
	static void AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta);

	template<>
	inline void AddEqualMatrix<MathDomain::Float>(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		if (aOperation != MatrixOperation::None)
			throw NotImplementedException();
		if (bOperation != MatrixOperation::None)
			throw NotImplementedException();

		cblas_sgeadd(columnMajorLayout,
							static_cast<int>(A.nRows), static_cast<int>(A.nCols),
							static_cast<float>(beta),
							reinterpret_cast<float*>(B.pointer), static_cast<int>(B.leadingDimension),
							static_cast<float>(alpha),
							reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension));
	}
	template<>
	inline void AddEqualMatrix<MathDomain::Double>(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		if (aOperation != MatrixOperation::None)
			throw NotImplementedException();
		if (bOperation != MatrixOperation::None)
			throw NotImplementedException();

		cblas_dgeadd(columnMajorLayout,
							static_cast<int>(A.nRows), static_cast<int>(A.nCols),
							beta,
							reinterpret_cast<double*>(B.pointer), static_cast<int>(B.leadingDimension),
							alpha,
							reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension));
	}

	template<MathDomain md>
	static void Scale(MemoryBuffer& z, const double alpha);

	template<>
	inline void Scale<MathDomain::Float>(MemoryBuffer& z, const double alpha)
	{
		cblas_sscal(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(z.pointer), 1);
	}
	template<>
	inline void Scale<MathDomain::Double>(MemoryBuffer& z, const double alpha)
	{
		cblas_dscal(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(z.pointer), 1);
	}

	template<MathDomain md>
	static void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha);

	template<>
	inline void ScaleColumns<MathDomain::Float>(MemoryTile& z, const MemoryBuffer& alpha)
	{
		for (size_t i = 0; i < z.nCols; ++i)
			cblas_sscal(static_cast<int>(z.nRows), *reinterpret_cast<float*>(alpha.pointer + i * alpha.ElementarySize()), reinterpret_cast<float*>(z.pointer + i * z.nRows * z.ElementarySize()), 1);
	}
	template<>
	inline void ScaleColumns<MathDomain::Double>(MemoryTile& z, const MemoryBuffer& alpha)
	{
		for (size_t i = 0; i < z.nCols; ++i)
			cblas_dscal(static_cast<int>(z.nRows), *reinterpret_cast<double*>(alpha.pointer + i * alpha.ElementarySize()), reinterpret_cast<double*>(z.pointer + i * z.nRows * z.ElementarySize()), 1);
	}

	template<MathDomain md>
	static void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta);

	template<>
	inline void SubMultiply<MathDomain::Float>(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		cblas_sgemm(columnMajorLayout, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(bOperation)], GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(cOperation)],
						 static_cast<int>(nRowsB), static_cast<int>(nColsC), static_cast<int>(nColsB),
						 static_cast<float>(alpha),
						 reinterpret_cast<float*>(B.pointer), static_cast<int>(B.leadingDimension),
						 reinterpret_cast<float*>(C.pointer), static_cast<int>(C.leadingDimension),
						 static_cast<float>(beta),
						 reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension));
	}

	template<>
	inline void SubMultiply<MathDomain::Double>(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		cblas_dgemm(columnMajorLayout, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(bOperation)], GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(cOperation)],
						 static_cast<int>(nRowsB), static_cast<int>(nColsC), static_cast<int>(nColsB),
						 alpha,
						 reinterpret_cast<double*>(B.pointer), static_cast<int>(B.leadingDimension),
						 reinterpret_cast<double*>(C.pointer), static_cast<int>(C.leadingDimension),
						 beta,
						 reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension));
	}

	template<MathDomain md>
	static void Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha = 1.0, const double beta = 0.0);

	template<>
	inline void Dot<MathDomain::Float>(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		cblas_sgemv(columnMajorLayout, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(aOperation)],
						 static_cast<int>(A.nRows), static_cast<int>(A.nCols),
						 static_cast<float>(alpha),
						 reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension),
						 reinterpret_cast<float*>(x.pointer), 1,
						 static_cast<float>(beta),
						 reinterpret_cast<float*>(y.pointer), 1);
	}
	template<>
	inline void Dot<MathDomain::Double>(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		cblas_dgemv(columnMajorLayout, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(aOperation)],
						 static_cast<int>(A.nRows), static_cast<int>(A.nCols),
						 alpha,
						 reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension),
						 reinterpret_cast<double*>(x.pointer), 1,
						 beta,
						 reinterpret_cast<double*>(y.pointer), 1);
	}

	template<MathDomain md>
	static void KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha);

	template<>
	inline void KroneckerProduct<MathDomain::Float>(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		cblas_sger(columnMajorLayout, static_cast<int>(x.size), static_cast<int>(y.size),
						static_cast<float>(alpha),
						reinterpret_cast<float*>(x.pointer), 1,
						reinterpret_cast<float*>(y.pointer), 1,
						reinterpret_cast<float*>(A.pointer), static_cast<int>(A.nRows));
	}
	template<>
	inline void KroneckerProduct<MathDomain::Double>(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		cblas_dger(columnMajorLayout, static_cast<int>(x.size), static_cast<int>(y.size),
						alpha,
						reinterpret_cast<double*>(x.pointer), 1,
						reinterpret_cast<double*>(y.pointer), 1,
						reinterpret_cast<double*>(A.pointer), static_cast<int>(A.nRows));
	}

	template<MathDomain md>
	static void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver);

	template<>
	inline void Solve<MathDomain::Float>(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
	{
		// Need to copy A, as it will be overwritten by its factorization
		MemoryTile aCopy(A);
		::cl::routines::Alloc(aCopy);
		Copy<MathDomain::Float>(aCopy, A);

		const auto nra = static_cast<int>(A.nRows);
		const auto lda = static_cast<int>(A.leadingDimension);
		const auto ncb = static_cast<int>(B.nCols);
		const auto ldb = static_cast<int>(B.leadingDimension);

		// Initializes auxliary value for solver
		int info = 0;

		switch (solver)
		{
			case LinearSystemSolverType::Lu:
			{
				// allocate memory for pivoting
				MemoryBuffer pivot(0, A.nRows, A.memorySpace, MathDomain::Int);
				::cl::routines::Alloc(pivot);

				// Factorize A (and overwrite it with L)
				info = LAPACKE_sgetrf(static_cast<int>(columnMajorLayout), nra, nra, reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<int*>(pivot.pointer));
				if (info != 0)
					throw OpenBlasException(__func__);

				// Solve factorized system
				info = LAPACKE_sgetrs(static_cast<int>(columnMajorLayout), openBlasOperation[static_cast<unsigned>(aOperation)], nra, ncb, reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<int*>(pivot.pointer), reinterpret_cast<float*>(B.pointer), ldb);
				if (info != 0)
					throw OpenBlasException(__func__);

				// free memory
				::cl::routines::Free(pivot);

				break;
			}
			case LinearSystemSolverType::Qr:
			{
				// allocate memory for tau
				MemoryBuffer tau(0, A.nRows, A.memorySpace, MathDomain::Float);
				::cl::routines::Alloc(tau);

				// A = Q * R
				/* int matrix_layout, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau */
				info = LAPACKE_sgeqrf(static_cast<int>(columnMajorLayout), nra, nra, reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<float*>(tau.pointer));
				if (info != 0)
					throw OpenBlasException(__func__);

				// B = Q^T * B
				/*
				 *  int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc */
				info = LAPACKE_sormqr(static_cast<int>(columnMajorLayout), 'L', 'T', 
											nra, nra, ncb,
											reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<float*>(tau.pointer), reinterpret_cast<float*>(B.pointer), ldb);
				if (info != 0)
					throw OpenBlasException(__func__);

				// Solve (x = R \ (Q^T * B))
				cblas_strsm(columnMajorLayout, CBLAS_SIDE::CblasLeft, CBLAS_UPLO::CblasUpper, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(aOperation)], CBLAS_DIAG::CblasNonUnit,
								 nra, nra, 1.0, reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<float*>(B.pointer), ldb);

				// free memory
				::cl::routines::Free(tau);

				break;
			}
			default:
				throw NotImplementedException();
		}

		::cl::routines::Free(aCopy);
	}

	template<>
	inline void Solve<MathDomain::Double>(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
	{
		// Need to copy A, as it will be overwritten by its factorization
		MemoryTile aCopy(A);
		::cl::routines::Alloc(aCopy);
		Copy<MathDomain::Double>(aCopy, A);

		const auto nra = static_cast<int>(A.nRows);
		const auto lda = static_cast<int>(A.leadingDimension);
		const auto ncb = static_cast<int>(B.nCols);
		const auto ldb = static_cast<int>(B.leadingDimension);

		// Initializes auxliary value for solver
		int info = 0;

		switch (solver)
		{
			case LinearSystemSolverType::Lu:
			{
				// allocate memory for pivoting
				MemoryBuffer pivot(0, A.nRows, A.memorySpace, MathDomain::Int);
				::cl::routines::Alloc(pivot);

				// Factorize A (and overwrite it with L)
				info = LAPACKE_dgetrf(static_cast<int>(columnMajorLayout), nra, nra, reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<int*>(pivot.pointer));
				if (info != 0)
					throw OpenBlasException(__func__);

				// Solve factorized system
				info = LAPACKE_dgetrs(static_cast<int>(columnMajorLayout), openBlasOperation[static_cast<unsigned>(aOperation)], nra, ncb, reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<int*>(pivot.pointer), reinterpret_cast<double*>(B.pointer), ldb);
				if (info != 0)
					throw OpenBlasException(__func__);

				// free memory
				::cl::routines::Free(pivot);

				break;
			}
			case LinearSystemSolverType::Qr:
			{
				// allocate memory for tau
				MemoryBuffer tau(0, A.nRows, A.memorySpace, MathDomain::Double);
				::cl::routines::Alloc(tau);

				// A = Q * R
				/* int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau */
				info = LAPACKE_dgeqrf(static_cast<int>(columnMajorLayout), nra, nra, reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<double*>(tau.pointer));
				if (info != 0)
					throw OpenBlasException(__func__);

				// B = Q^T * B
				/*
				 *  int matrix_layout, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc */
				info = LAPACKE_dormqr(static_cast<int>(columnMajorLayout), 'L', 'T',
											 nra, nra, ncb,
											 reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<double*>(tau.pointer), reinterpret_cast<double*>(B.pointer), ldb);
				if (info != 0)
					throw OpenBlasException(__func__);

				// Solve (x = R \ (Q^T * B))
				cblas_dtrsm(columnMajorLayout, CBLAS_SIDE::CblasLeft, CBLAS_UPLO::CblasUpper, GENERIC_API_NAMESPACEOperationsEnum[static_cast<unsigned>(aOperation)], CBLAS_DIAG::CblasNonUnit,
										  nra, nra, 1.0, reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<double*>(B.pointer), ldb);

				// free memory
				::cl::routines::Free(tau);

				break;
			}
			default:
				throw NotImplementedException();
		}

		::cl::routines::Free(aCopy);
	}

	template<MathDomain md>
	static void ArgAbsMin(int& argMin, const MemoryBuffer& x);

	template<>
	inline void ArgAbsMin<MathDomain::Float>(int&, const MemoryBuffer& )
	{
		// TODO: apparently there's isamax but not isamin??
		//argMin = static_cast<int>(cblas_isamin(static_cast<int>(x.size), reinterpret_cast<float*>(x.pointer), 1));
	}
	template<>
	inline void ArgAbsMin<MathDomain::Double>(int&, const MemoryBuffer& )
	{
		// TODO: apparently there's isamax but not isamin??
		//argMin = static_cast<int>(cblas_idamin(static_cast<int>(x.size), reinterpret_cast<double*>(x.pointer), 1));
	}

	template<MathDomain md>
	static void ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A);

	template<>
	inline void ColumnWiseArgAbsMin<MathDomain::Float>(MemoryBuffer&, const MemoryTile&)
	{
		// TODO
//		auto* argMinPtr = reinterpret_cast<int*>(argMin.pointer);
//		// 1 added for compatibility
//		for (size_t j = 0; j < A.nCols; ++j)
//			argMinPtr[j] = 1 + static_cast<int>(cblas_isamin(static_cast<int>(A.nRows), reinterpret_cast<float*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
	}
	template<>
	inline void ColumnWiseArgAbsMin<MathDomain::Double>(MemoryBuffer&, const MemoryTile&)
	{
		// TODO
//		// 1 added for compatibility
//		auto* argMinPtr = reinterpret_cast<int*>(argMin.pointer);
//		for (size_t j = 0; j < A.nCols; ++j)
//			argMinPtr[j] = 1 + static_cast<int>(cblas_idamin(static_cast<int>(A.nRows), reinterpret_cast<double*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
	}

	template<MathDomain md>
	static void ArgAbsMax(int& argMax, const MemoryBuffer& x);

	template<>
	inline void ArgAbsMax<MathDomain::Float>(int& argMax, const MemoryBuffer& x)
	{
		argMax = static_cast<int>(cblas_isamax(static_cast<int>(x.size), reinterpret_cast<float*>(x.pointer), 1));
	}
	template<>
	inline void ArgAbsMax<MathDomain::Double>(int& argMax, const MemoryBuffer& x)
	{
		argMax = static_cast<int>(cblas_idamax(static_cast<int>(x.size), reinterpret_cast<double*>(x.pointer), 1));
	}

	template<MathDomain md>
	static void ColumnWiseArgAbsMax(MemoryBuffer& argMax, const MemoryTile& A);

	template<>
	inline void ColumnWiseArgAbsMax<MathDomain::Float>(MemoryBuffer& argMax, const MemoryTile& A)
	{
		// 1 added for compatibility
		auto* argMaxPtr = reinterpret_cast<int*>(argMax.pointer);
		for (size_t j = 0; j < A.nCols; ++j)
			argMaxPtr[j] = 1 + static_cast<int>(cblas_isamax(static_cast<int>(A.nRows), reinterpret_cast<float*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
	}
	template<>
	inline void ColumnWiseArgAbsMax<MathDomain::Double>(MemoryBuffer& argMax, const MemoryTile& A)
	{
		// 1 added for compatibility
		auto* argMaxPtr = reinterpret_cast<int*>(argMax.pointer);
		for (size_t j = 0; j < A.nCols; ++j)
			argMaxPtr[j] = 1 + static_cast<int>(cblas_idamax(static_cast<int>(A.nRows), reinterpret_cast<double*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
	}

	// norm = ||x||_2
	template<MathDomain md>
	static void EuclideanNorm(double& norm, const MemoryBuffer& z);

	template<>
	inline void EuclideanNorm<MathDomain::Float>(double& norm, const MemoryBuffer& z)
	{
		norm = static_cast<double>(cblas_snrm2(static_cast<int>(z.size), reinterpret_cast<float*>(z.pointer), 1));
	}
	template<>
	inline void EuclideanNorm<MathDomain::Double>(double& norm, const MemoryBuffer& z)
	{
		norm = cblas_dnrm2(static_cast<int>(z.size), reinterpret_cast<double*>(z.pointer), 1);
	}
}}}

#endif

#undef GENERIC_API_DEFINE
#undef GENERIC_API_NAMESPACE
#undef GENERIC_API_ROUTINES_NAMESPACE
#undef ROUTINES_NAMESPACE
#undef BLAS_NAMESPACE
