#pragma once

#include <Types.h>
#include <BufferInitializer.h>

#include <vector>
#include <array>

#ifndef USE_MKL

		template<MathDomain md>
		static void SparseAdd(MemoryBuffer&, const SparseMemoryBuffer&, const MemoryBuffer&, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void SparseDot(MemoryBuffer&, const SparseMemoryTile&, const MemoryBuffer&, const MatrixOperation, const double, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void SparseMultiply(MemoryTile&, const SparseMemoryTile&, const MemoryTile&, const MatrixOperation, const double)
		{
			throw NotImplementedException();
		}

		template<MathDomain md>
		static void SparseSolve(SparseMemoryTile&, MemoryTile&, LinearSystemSolverType)
		{
			throw NotImplementedException();
		}

#else

	#include <cmath>
	namespace mkl
	{
		#include <mkl.h>
		#include <mkl_sparse_qr.h>
	}

	namespace cl { namespace routines { namespace mkr {
		static constexpr std::array<mkl::sparse_operation_t, 2> mklSparseOperations { mkl::sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE, mkl::sparse_operation_t::SPARSE_OPERATION_TRANSPOSE  };

		template<MathDomain md>
		static void AllocateCsrHandle(SparseMemoryTile& A);
		template<>
		inline void AllocateCsrHandle<MathDomain::Float>(SparseMemoryTile& A)
		{
			mkl::sparse_matrix_t sparseA;
			mkl::mkl_sparse_s_create_csr(&sparseA,
										 mkl::sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
										 static_cast<int>(A.nRows),
										 static_cast<int>(A.nCols),
										 reinterpret_cast<int*>(A.nNonZeroRows),
										 reinterpret_cast<int*>(A.nNonZeroRows) + 1,
										 reinterpret_cast<int*>(A.nonZeroColumnIndices),
										 reinterpret_cast<float*>(A.pointer));

			A.thirdPartyHandle = reinterpret_cast<ptr_t>(sparseA);
		}

		template<>
		inline void AllocateCsrHandle<MathDomain::Double>(SparseMemoryTile& A)
		{
			mkl::sparse_matrix_t sparseA;
			mkl::mkl_sparse_d_create_csr(&sparseA,
										 mkl::sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
										 static_cast<int>(A.nRows),
										 static_cast<int>(A.nCols),
										 reinterpret_cast<int*>(A.nNonZeroRows),
										 reinterpret_cast<int*>(A.nNonZeroRows) + 1,
										 reinterpret_cast<int*>(A.nonZeroColumnIndices),
										 reinterpret_cast<double*>(A.pointer));

			A.thirdPartyHandle = reinterpret_cast<ptr_t>(sparseA);
		}

		template<MathDomain md>
		static inline void DestroyCsrHandle(SparseMemoryTile& A)
		{
			mkl::mkl_sparse_destroy(reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle));
		}

		template<MathDomain md>
		static void SparseAdd(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha);
		template<>
		inline void SparseAdd<MathDomain::Float>(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
		{
			Copy<MathDomain::Float>(z, y);
			mkl::cblas_saxpyi(static_cast<int>(x.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), reinterpret_cast<int*>(x.indices), reinterpret_cast<float*>(z.pointer));
		}
		template<>
		inline void SparseAdd<MathDomain::Double>(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
		{
			Copy<MathDomain::Double>(z, y);
			mkl::cblas_daxpyi(static_cast<int>(x.size), alpha, reinterpret_cast<double*>(x.pointer), reinterpret_cast<int*>(x.indices), reinterpret_cast<double*>(z.pointer));
		}

		template<MathDomain md>
		static void SparseDot(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta);
		template<>
		inline void SparseDot<MathDomain::Float>(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
		{
			if (A.thirdPartyHandle == 0)
				AllocateCsrHandle<MathDomain::Float>(A);

			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;

			mkl::mkl_sparse_s_mv(mklSparseOperations[static_cast<size_t>(aOperation)],
								 static_cast<float>(alpha),
								 reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), descr,
								 reinterpret_cast<float*>(x.pointer),
								 static_cast<float>(beta), reinterpret_cast<float*>(y.pointer));
		}
		template<>
		inline void SparseDot<MathDomain::Double>(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
		{
			if (A.thirdPartyHandle == 0)
				AllocateCsrHandle<MathDomain::Double>(A);

			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_d_mv(mklSparseOperations[static_cast<size_t>(aOperation)],
								 alpha,
								 reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), descr,
								 reinterpret_cast<double*>(x.pointer),
								 beta, reinterpret_cast<double*>(y.pointer));
		}

		template<MathDomain md>
		static void SparseMultiply(MemoryTile& A, SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha);
		template<>
		inline void SparseMultiply<MathDomain::Float>(MemoryTile& A, SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
		{
			if (B.thirdPartyHandle == 0)
				AllocateCsrHandle<MathDomain::Float>(B);

			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_s_mm(mklSparseOperations[static_cast<size_t>(bOperation)],
								 static_cast<float>(alpha),
								 reinterpret_cast<mkl::sparse_matrix_t>(B.thirdPartyHandle), descr,
								 mkl::sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,
								 reinterpret_cast<float*>(C.pointer), static_cast<int>(A.nCols), static_cast<int>(C.leadingDimension),
								 0.0f, reinterpret_cast<float*>(A.pointer), static_cast<int>(A.leadingDimension));
		}
		template<>
		inline void SparseMultiply<MathDomain::Double>(MemoryTile& A, SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
		{
			if (B.thirdPartyHandle == 0)
				AllocateCsrHandle<MathDomain::Double>(B);

			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_d_mm(mklSparseOperations[static_cast<size_t>(bOperation)],
								 alpha,
								 reinterpret_cast<mkl::sparse_matrix_t>(B.thirdPartyHandle), descr,
								 mkl::sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,
								 reinterpret_cast<double*>(C.pointer), static_cast<int>(A.nCols), static_cast<int>(A.leadingDimension),
								 0.0, reinterpret_cast<double*>(A.pointer), static_cast<int>(A.leadingDimension));
		}

		template<MathDomain md>
		static void SparseSolve(SparseMemoryTile& A, MemoryTile& B, LinearSystemSolverType solver);
		template<>
		inline void SparseSolve<MathDomain::Float>(SparseMemoryTile& A, MemoryTile& B, LinearSystemSolverType solver)
		{
			SparseMemoryTile aCopy(A);
			aCopy.pointer = 0;
			Alloc(aCopy);

			switch (solver)
			{
				case LinearSystemSolverType::Qr:
				{
					if (A.thirdPartyHandle == 0)
						AllocateCsrHandle<MathDomain::Float>(A);

					mkl::matrix_descr descr{};
					descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
					descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
					descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
					auto err = mkl::mkl_sparse_qr_reorder(reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), descr);
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);

					err = mkl::mkl_sparse_s_qr_factorize(reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), reinterpret_cast<float*>(aCopy.pointer));
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);

					err = mkl::mkl_sparse_s_qr_solve(mkl::SPARSE_OPERATION_NON_TRANSPOSE,
											   reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle),
											   reinterpret_cast<float *>(aCopy.pointer),
											   mkl::SPARSE_LAYOUT_COLUMN_MAJOR, static_cast<int>(B.nCols),
											   reinterpret_cast<float *>(B.pointer), static_cast<int>(B.leadingDimension),
											   reinterpret_cast<float *>(B.pointer), static_cast<int>(B.leadingDimension));
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);

					break;
				}
				default:
					throw NotImplementedException();
			}

			Free(aCopy);
		}

		template<>
		inline void SparseSolve<MathDomain::Double>(SparseMemoryTile& A, MemoryTile& B, LinearSystemSolverType solver)
		{
			SparseMemoryTile aCopy(A);
			aCopy.pointer = 0;
			Alloc(aCopy);

			switch (solver)
			{
				case LinearSystemSolverType::Qr:
				{
					if (A.thirdPartyHandle == 0)
						AllocateCsrHandle<MathDomain::Double>(A);

					mkl::matrix_descr descr{};
					descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
					descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
					descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
					auto err = mkl::mkl_sparse_qr_reorder(reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), descr);
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);
					err = mkl::mkl_sparse_d_qr_factorize(reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle), reinterpret_cast<double *>(aCopy.pointer));
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);

					err = mkl::mkl_sparse_d_qr_solve(mkl::SPARSE_OPERATION_NON_TRANSPOSE,
											   reinterpret_cast<mkl::sparse_matrix_t>(A.thirdPartyHandle),
											   reinterpret_cast<double *>(aCopy.pointer),
											   mkl::SPARSE_LAYOUT_COLUMN_MAJOR, 1,
											   reinterpret_cast<double *>(B.pointer), static_cast<int>(B.leadingDimension),
											   reinterpret_cast<double *>(B.pointer), static_cast<int>(B.leadingDimension));
					if (err != mkl::SPARSE_STATUS_SUCCESS)
						throw MklException(__func__);

					break;
				}
				default:
					throw NotImplementedException();
			}

			Free(aCopy);
		}

	}}}

#endif
