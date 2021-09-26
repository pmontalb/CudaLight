#pragma once

#include <BufferInitializer.h>
#include <Types.h>
#include <Exceptions.h>

#include <array>
#include <vector>

#ifndef USE_MKL

namespace cl
{
	namespace routines
	{
		namespace mkr
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
		}	 // namespace mkr
	}		 // namespace routines
}	 // namespace cl

#else

	#include <cmath>
namespace mkl
{
	#include <mkl.h>
}

namespace cl
{
	namespace routines
	{
		namespace mkr
		{
			static constexpr mkl::CBLAS_LAYOUT columnMajorLayout = { mkl::CBLAS_LAYOUT::CblasColMajor };
			static constexpr std::array<mkl::CBLAS_TRANSPOSE, 2> mklOperationsEnum = { mkl::CBLAS_TRANSPOSE::CblasNoTrans, mkl::CBLAS_TRANSPOSE::CblasTrans };

			template<MathDomain md>
			static void Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha);

			template<>
			inline void Add<MathDomain::Float>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				Copy<MathDomain::Float>(z, y);
				mkl::cblas_saxpy(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), 1, reinterpret_cast<float*>(z.pointer), 1);
			}
			template<>
			inline void Add<MathDomain::Double>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				Copy<MathDomain::Double>(z, y);
				mkl::cblas_daxpy(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(x.pointer), 1, reinterpret_cast<double*>(z.pointer), 1);
			}

			template<MathDomain md>
			static void AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha);

			template<>
			inline void AddEqual<MathDomain::Float>(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
			{
				mkl::cblas_saxpy(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), 1, reinterpret_cast<float*>(z.pointer), 1);
			}
			template<>
			inline void AddEqual<MathDomain::Double>(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
			{
				mkl::cblas_daxpy(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(x.pointer), 1, reinterpret_cast<double*>(z.pointer), 1);
			}

			template<MathDomain md>
			static void AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta);

			template<>
			inline void AddEqualMatrix<MathDomain::Float>(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
			{
				mkl::MKL_Somatadd(columnMajorOrdering, mklOperation[static_cast<unsigned>(aOperation)], mklOperation[static_cast<unsigned>(bOperation)], A.nRows, A.nCols, static_cast<float>(alpha), reinterpret_cast<float*>(A.pointer), A.leadingDimension, static_cast<float>(beta), reinterpret_cast<float*>(B.pointer), B.leadingDimension, reinterpret_cast<float*>(A.pointer), A.leadingDimension);
			}
			template<>
			inline void AddEqualMatrix<MathDomain::Double>(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
			{
				mkl::MKL_Domatadd(columnMajorOrdering, mklOperation[static_cast<unsigned>(aOperation)], mklOperation[static_cast<unsigned>(bOperation)], A.nRows, A.nCols, alpha, reinterpret_cast<double*>(A.pointer), A.leadingDimension, beta, reinterpret_cast<double*>(B.pointer), B.leadingDimension, reinterpret_cast<double*>(A.pointer), A.leadingDimension);
			}

			template<MathDomain md>
			static void Scale(MemoryBuffer& z, const double alpha);

			template<>
			inline void Scale<MathDomain::Float>(MemoryBuffer& z, const double alpha)
			{
				mkl::cblas_sscal(static_cast<int>(z.size), static_cast<float>(alpha), reinterpret_cast<float*>(z.pointer), 1);
			}
			template<>
			inline void Scale<MathDomain::Double>(MemoryBuffer& z, const double alpha)
			{
				mkl::cblas_dscal(static_cast<int>(z.size), alpha, reinterpret_cast<double*>(z.pointer), 1);
			}

			template<MathDomain md>
			static void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha);

			template<>
			inline void ScaleColumns<MathDomain::Float>(MemoryTile& z, const MemoryBuffer& alpha)
			{
				for (size_t i = 0; i < z.nCols; ++i)
					mkl::cblas_sscal(static_cast<int>(z.nRows), *reinterpret_cast<float*>(alpha.pointer + i * alpha.ElementarySize()), reinterpret_cast<float*>(z.pointer + i * z.nRows * z.ElementarySize()), 1);
			}
			template<>
			inline void ScaleColumns<MathDomain::Double>(MemoryTile& z, const MemoryBuffer& alpha)
			{
				for (size_t i = 0; i < z.nCols; ++i)
					mkl::cblas_dscal(static_cast<int>(z.nRows), *reinterpret_cast<double*>(alpha.pointer + i * alpha.ElementarySize()), reinterpret_cast<double*>(z.pointer + i * z.nRows * z.ElementarySize()), 1);
			}

			template<MathDomain md>
			static void ElementwiseProduct(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha);

			template<>
			inline void ElementwiseProduct<MathDomain::Float>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				mkl::vsMul(static_cast<int>(z.size), reinterpret_cast<float*>(x.pointer), reinterpret_cast<float*>(y.pointer), reinterpret_cast<float*>(z.pointer));
				if (std::fabs(alpha - 1.0) > 1e-7)
					Scale<MathDomain::Float>(z, alpha);
			}

			template<>
			inline void ElementwiseProduct<MathDomain::Double>(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				mkl::vdMul(static_cast<int>(z.size), reinterpret_cast<double*>(x.pointer), reinterpret_cast<double*>(y.pointer), reinterpret_cast<double*>(z.pointer));
				if (std::fabs(alpha - 1.0) > 1e-7)
					Scale<MathDomain::Float>(z, alpha);
			}

			template<MathDomain md>
			static void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta);

			template<>
			inline void SubMultiply<MathDomain::Float>(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
			{
				mkl::cblas_sgemm(columnMajorLayout, mklOperationsEnum[static_cast<unsigned>(bOperation)], mklOperationsEnum[static_cast<unsigned>(cOperation)], static_cast<int>(nRowsB), static_cast<int>(nColsC), static_cast<int>(nColsB), static_cast<float>(alpha), reinterpret_cast<float*>(B.pointer), static_cast<int>(B.leadingDimension), reinterpret_cast<float*>(C.pointer), static_cast<int>(C.leadingDimension), static_cast<float>(beta), reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension));
			}

			template<>
			inline void SubMultiply<MathDomain::Double>(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
			{
				mkl::cblas_dgemm(columnMajorLayout, mklOperationsEnum[static_cast<unsigned>(bOperation)], mklOperationsEnum[static_cast<unsigned>(cOperation)], static_cast<int>(nRowsB), static_cast<int>(nColsC), static_cast<int>(nColsB), alpha, reinterpret_cast<double*>(B.pointer), static_cast<int>(B.leadingDimension), reinterpret_cast<double*>(C.pointer), static_cast<int>(C.leadingDimension), beta, reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension));
			}

			template<MathDomain md>
			static void BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta);

			template<>
			inline void BatchedMultiply<MathDomain::Float>(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
			{
				const auto nGroups = A.nCubes;
				std::vector<mkl::CBLAS_TRANSPOSE> bOperations(nGroups, mklOperationsEnum[static_cast<unsigned>(bOperation)]);
				std::vector<mkl::CBLAS_TRANSPOSE> cOperations(nGroups, mklOperationsEnum[static_cast<unsigned>(cOperation)]);
				std::vector<int> nRowsA(nGroups, static_cast<int>(A.nRows));
				std::vector<int> nColsA(nGroups, static_cast<int>(A.nCols));
				std::vector<int> nColsB(nGroups, static_cast<int>(B.nCols));
				std::vector<float> alphas(nGroups, static_cast<float>(alpha));
				std::vector<float> betas(nGroups, static_cast<float>(beta));
				std::vector<int> lda(nGroups, static_cast<int>(A.leadingDimension));
				std::vector<int> ldb(nGroups, static_cast<int>(B.leadingDimension));
				std::vector<int> ldc(nGroups, static_cast<int>(C.leadingDimension));
				std::vector<int> groupSizes(nGroups, 1);

				std::vector<const float*> bPointers(nGroups);
				std::vector<const float*> cPointers(nGroups);
				std::vector<float*> aPointers(nGroups);
				for (size_t i = 0; i < nGroups; ++i)
				{
					bPointers[i] = reinterpret_cast<float*>(B.pointer + i * strideB * B.ElementarySize());
					cPointers[i] = reinterpret_cast<float*>(C.pointer + i * strideC * C.ElementarySize());
					aPointers[i] = reinterpret_cast<float*>(A.pointer + i * A.nRows * A.nCols * A.ElementarySize());
				}
				mkl::cblas_sgemm_batch(columnMajorLayout, bOperations.data(), cOperations.data(), nRowsA.data(), nColsA.data(), nColsB.data(), alphas.data(), bPointers.data(), ldb.data(), cPointers.data(), ldc.data(), betas.data(), aPointers.data(), lda.data(), static_cast<int>(nGroups), groupSizes.data());
			}
			template<>
			inline void BatchedMultiply<MathDomain::Double>(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
			{
				const auto nGroups = A.nCubes;
				std::vector<mkl::CBLAS_TRANSPOSE> bOperations(nGroups, mklOperationsEnum[static_cast<unsigned>(bOperation)]);
				std::vector<mkl::CBLAS_TRANSPOSE> cOperations(nGroups, mklOperationsEnum[static_cast<unsigned>(cOperation)]);
				std::vector<int> nRowsA(nGroups, static_cast<int>(A.nRows));
				std::vector<int> nColsA(nGroups, static_cast<int>(A.nCols));
				std::vector<int> nColsB(nGroups, static_cast<int>(B.nCols));
				std::vector<double> alphas(nGroups, alpha);
				std::vector<double> betas(nGroups, beta);
				std::vector<int> lda(nGroups, static_cast<int>(A.leadingDimension));
				std::vector<int> ldb(nGroups, static_cast<int>(B.leadingDimension));
				std::vector<int> ldc(nGroups, static_cast<int>(C.leadingDimension));
				std::vector<int> groupSizes(nGroups, 1);

				std::vector<const double*> bPointers(nGroups);
				std::vector<const double*> cPointers(nGroups);
				std::vector<double*> aPointers(nGroups);
				for (size_t i = 0; i < nGroups; ++i)
				{
					bPointers[i] = reinterpret_cast<double*>(B.pointer + i * strideB * B.ElementarySize());
					cPointers[i] = reinterpret_cast<double*>(C.pointer + i * strideC * C.ElementarySize());
					aPointers[i] = reinterpret_cast<double*>(A.pointer + i * A.nRows * A.nCols * A.ElementarySize());
				}
				mkl::cblas_dgemm_batch(columnMajorLayout, bOperations.data(), cOperations.data(), nRowsA.data(), nColsA.data(), nColsB.data(), alphas.data(), bPointers.data(), ldb.data(), cPointers.data(), ldc.data(), betas.data(), aPointers.data(), lda.data(), static_cast<int>(nGroups), groupSizes.data());
			}

			template<MathDomain md>
			static void Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha = 1.0, const double beta = 0.0);

			template<>
			inline void Dot<MathDomain::Float>(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
			{
				mkl::cblas_sgemv(columnMajorLayout, mklOperationsEnum[static_cast<unsigned>(aOperation)], static_cast<int>(A.nRows), static_cast<int>(A.nCols), static_cast<float>(alpha), reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension), reinterpret_cast<float*>(x.pointer), 1, static_cast<float>(beta), reinterpret_cast<float*>(y.pointer), 1);
			}
			template<>
			inline void Dot<MathDomain::Double>(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
			{
				mkl::cblas_dgemv(columnMajorLayout, mklOperationsEnum[static_cast<unsigned>(aOperation)], static_cast<int>(A.nRows), static_cast<int>(A.nCols), alpha, reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension), reinterpret_cast<double*>(x.pointer), 1, beta, reinterpret_cast<double*>(y.pointer), 1);
			}

			template<MathDomain md>
			static void KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha);

			template<>
			inline void KroneckerProduct<MathDomain::Float>(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				mkl::cblas_sger(columnMajorLayout, static_cast<int>(x.size), static_cast<int>(y.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), 1, reinterpret_cast<float*>(y.pointer), 1, reinterpret_cast<float*>(A.pointer), static_cast<int>(A.nRows));
			}
			template<>
			inline void KroneckerProduct<MathDomain::Double>(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
			{
				mkl::cblas_dger(columnMajorLayout, static_cast<int>(x.size), static_cast<int>(y.size), alpha, reinterpret_cast<double*>(x.pointer), 1, reinterpret_cast<double*>(y.pointer), 1, reinterpret_cast<double*>(A.pointer), static_cast<int>(A.nRows));
			}

			template<MathDomain md>
			static void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver);

			template<>
			inline void Solve<MathDomain::Float>(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
			{
				// Need to copy A, as it will be overwritten by its factorization
				MemoryTile aCopy(A);
				Alloc(aCopy);
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
						Alloc(pivot);

						// Factorize A (and overwrite it with L)
						mkl::sgetrf(&nra, &nra, reinterpret_cast<float*>(aCopy.pointer), &lda, reinterpret_cast<int*>(pivot.pointer), &info);
						if (info != 0)
							throw MklException(__func__);

						// Solve factorized system
						mkl::sgetrs(mklOperationGemm[static_cast<unsigned>(aOperation)], &nra, &ncb, reinterpret_cast<float*>(aCopy.pointer), &lda, reinterpret_cast<int*>(pivot.pointer), reinterpret_cast<float*>(B.pointer), &ldb, &info);
						if (info != 0)
							throw MklException(__func__);

						// free memory
						Free(pivot);

						break;
					}
					case LinearSystemSolverType::Qr:
					{
						// allocate memory for tau
						MemoryBuffer tau(0, A.nRows, A.memorySpace, MathDomain::Float);
						Alloc(tau);

						static constexpr size_t workBufferMultiple = { 64 };
						const int workSize = static_cast<int>(A.nCols * workBufferMultiple);
						// allocate memory for workBuffer
						MemoryBuffer buffer(0, static_cast<unsigned>(workSize), A.memorySpace, MathDomain::Float);
						Alloc(buffer);

						// A = Q * R
						mkl::sgeqrf(&nra, &nra, reinterpret_cast<float*>(aCopy.pointer), &lda, reinterpret_cast<float*>(tau.pointer), reinterpret_cast<float*>(buffer.pointer), &workSize, &info);
						if (info != 0)
							throw MklException(__func__);

						// B = Q^T * B
						mkl::sormqr("L", "T", &nra, &nra, &ncb, reinterpret_cast<float*>(aCopy.pointer), &lda, reinterpret_cast<float*>(tau.pointer), reinterpret_cast<float*>(B.pointer), &ldb, reinterpret_cast<float*>(buffer.pointer), &workSize, &info);
						if (info != 0)
							throw MklException(__func__);

						// Solve (x = R \ (Q^T * B))
						mkl::cblas_strsm(columnMajorLayout, mkl::CBLAS_SIDE::CblasLeft, mkl::CBLAS_UPLO::CblasUpper, mklOperationsEnum[static_cast<unsigned>(aOperation)], mkl::CBLAS_DIAG::CblasNonUnit, nra, nra, 1.0, reinterpret_cast<float*>(aCopy.pointer), lda, reinterpret_cast<float*>(B.pointer), ldb);

						// free memory
						Free(tau);
						Free(buffer);

						break;
					}
					default:
						throw NotImplementedException();
				}

				Free(aCopy);
			}

			template<>
			inline void Solve<MathDomain::Double>(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
			{
				// Need to copy A, as it will be overwritten by its factorization
				MemoryTile aCopy(A);
				Alloc(aCopy);
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
						Alloc(pivot);

						// Factorize A (and overwrite it with L)
						mkl::dgetrf(&nra, &nra, reinterpret_cast<double*>(aCopy.pointer), &lda, reinterpret_cast<int*>(pivot.pointer), &info);
						if (info != 0)
							throw MklException(__func__);

						// Solve factorized system
						mkl::dgetrs(mklOperationGemm[static_cast<unsigned>(aOperation)], &nra, &ncb, reinterpret_cast<double*>(aCopy.pointer), &lda, reinterpret_cast<int*>(pivot.pointer), reinterpret_cast<double*>(B.pointer), &ldb, &info);
						if (info != 0)
							throw MklException(__func__);

						// free memory
						Free(pivot);

						break;
					}
					case LinearSystemSolverType::Qr:
					{
						// allocate memory for tau
						MemoryBuffer tau(0, A.nRows, A.memorySpace, MathDomain::Double);
						Alloc(tau);

						static constexpr size_t workBufferMultiple = { 64 };
						const int workSize = static_cast<int>(A.nCols * workBufferMultiple);
						// allocate memory for workBuffer
						MemoryBuffer buffer(0, static_cast<unsigned>(workSize), A.memorySpace, MathDomain::Double);
						Alloc(buffer);

						// A = Q * R
						mkl::dgeqrf(&nra, &nra, reinterpret_cast<double*>(aCopy.pointer), &lda, reinterpret_cast<double*>(tau.pointer), reinterpret_cast<double*>(buffer.pointer), &workSize, &info);
						if (info != 0)
							throw MklException(__func__);

						// B = Q^T * B
						mkl::dormqr("L", "T", &nra, &nra, &ncb, reinterpret_cast<double*>(aCopy.pointer), &lda, reinterpret_cast<double*>(tau.pointer), reinterpret_cast<double*>(B.pointer), &ldb, reinterpret_cast<double*>(buffer.pointer), &workSize, &info);
						if (info != 0)
							throw MklException(__func__);

						// Solve (x = R \ (Q^T * B))
						mkl::cblas_dtrsm(columnMajorLayout, mkl::CBLAS_SIDE::CblasLeft, mkl::CBLAS_UPLO::CblasUpper, mklOperationsEnum[static_cast<unsigned>(aOperation)], mkl::CBLAS_DIAG::CblasNonUnit, nra, nra, 1.0, reinterpret_cast<double*>(aCopy.pointer), lda, reinterpret_cast<double*>(B.pointer), ldb);

						break;
					}
					default:
						throw NotImplementedException();
				}

				Free(aCopy);
			}

			template<MathDomain md>
			static void ArgAbsMin(int& argMin, const MemoryBuffer& x);

			template<>
			inline void ArgAbsMin<MathDomain::Float>(int& argMin, const MemoryBuffer& x)
			{
				argMin = static_cast<int>(mkl::cblas_isamin(static_cast<int>(x.size), reinterpret_cast<float*>(x.pointer), 1));
			}
			template<>
			inline void ArgAbsMin<MathDomain::Double>(int& argMin, const MemoryBuffer& x)
			{
				argMin = static_cast<int>(mkl::cblas_idamin(static_cast<int>(x.size), reinterpret_cast<double*>(x.pointer), 1));
			}

			template<MathDomain md>
			static void ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A);

			template<>
			inline void ColumnWiseArgAbsMin<MathDomain::Float>(MemoryBuffer& argMin, const MemoryTile& A)
			{
				auto* argMinPtr = reinterpret_cast<int*>(argMin.pointer);
				// 1 added for compatibility
				for (size_t j = 0; j < A.nCols; ++j)
					argMinPtr[j] = 1 + static_cast<int>(mkl::cblas_isamin(static_cast<int>(A.nRows), reinterpret_cast<float*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
			}
			template<>
			inline void ColumnWiseArgAbsMin<MathDomain::Double>(MemoryBuffer& argMin, const MemoryTile& A)
			{
				// 1 added for compatibility
				auto* argMinPtr = reinterpret_cast<int*>(argMin.pointer);
				for (size_t j = 0; j < A.nCols; ++j)
					argMinPtr[j] = 1 + static_cast<int>(mkl::cblas_idamin(static_cast<int>(A.nRows), reinterpret_cast<double*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
			}

			template<MathDomain md>
			static void ArgAbsMax(int& argMax, const MemoryBuffer& x);

			template<>
			inline void ArgAbsMax<MathDomain::Float>(int& argMax, const MemoryBuffer& x)
			{
				argMax = static_cast<int>(mkl::cblas_isamax(static_cast<int>(x.size), reinterpret_cast<float*>(x.pointer), 1));
			}
			template<>
			inline void ArgAbsMax<MathDomain::Double>(int& argMax, const MemoryBuffer& x)
			{
				argMax = static_cast<int>(mkl::cblas_idamax(static_cast<int>(x.size), reinterpret_cast<double*>(x.pointer), 1));
			}

			template<MathDomain md>
			static void ColumnWiseArgAbsMax(MemoryBuffer& argMax, const MemoryTile& A);

			template<>
			inline void ColumnWiseArgAbsMax<MathDomain::Float>(MemoryBuffer& argMax, const MemoryTile& A)
			{
				// 1 added for compatibility
				auto* argMaxPtr = reinterpret_cast<int*>(argMax.pointer);
				for (size_t j = 0; j < A.nCols; ++j)
					argMaxPtr[j] = 1 + static_cast<int>(mkl::cblas_isamax(static_cast<int>(A.nRows), reinterpret_cast<float*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
			}
			template<>
			inline void ColumnWiseArgAbsMax<MathDomain::Double>(MemoryBuffer& argMax, const MemoryTile& A)
			{
				// 1 added for compatibility
				auto* argMaxPtr = reinterpret_cast<int*>(argMax.pointer);
				for (size_t j = 0; j < A.nCols; ++j)
					argMaxPtr[j] = 1 + static_cast<int>(mkl::cblas_idamax(static_cast<int>(A.nRows), reinterpret_cast<double*>(A.pointer + j * A.nRows * A.ElementarySize()), 1));
			}

			// norm = ||x||_2
			template<MathDomain md>
			static void EuclideanNorm(double& norm, const MemoryBuffer& z);

			template<>
			inline void EuclideanNorm<MathDomain::Float>(double& norm, const MemoryBuffer& z)
			{
				norm = static_cast<double>(mkl::cblas_snrm2(static_cast<int>(z.size), reinterpret_cast<float*>(z.pointer), 1));
			}
			template<>
			inline void EuclideanNorm<MathDomain::Double>(double& norm, const MemoryBuffer& z)
			{
				norm = mkl::cblas_dnrm2(static_cast<int>(z.size), reinterpret_cast<double*>(z.pointer), 1);
			}
		}	 // namespace mkr
	}		 // namespace routines
}	 // namespace cl

#endif
