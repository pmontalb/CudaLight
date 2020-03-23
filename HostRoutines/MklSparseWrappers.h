#pragma once

#include <Types.h>
#include <BufferInitializer.h>

#include <vector>
#include <array>

#ifndef USE_MKL

	template<MathDomain md>
	static void Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void Scale(MemoryBuffer& z, const double alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ElementwiseProduct(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha = 1.0, const double beta = 0.0)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ArgAbsMin(int& argMin, const MemoryBuffer& x)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ArgAbsMax(int& argMax, const MemoryBuffer& x)
	{
		throw NotImplementedException();
	}

	template<MathDomain md>
	static void ColumnWiseArgAbsMax(MemoryBuffer& argMax, const MemoryTile& A)
	{
		throw NotImplementedException();
	}

	// norm = ||x||_2
	template<MathDomain md>
	static void EuclideanNorm(double& norm, const MemoryBuffer& z)
	{
		throw NotImplementedException();
	}

#else

	#include <cmath>
	namespace mkl
	{
		#include <mkl.h>
	}

	namespace cl { namespace routines { namespace mkr {
		static constexpr std::array<mkl::sparse_operation_t, 2> mklSparseOperations { mkl::sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE, mkl::sparse_operation_t::SPARSE_OPERATION_TRANSPOSE  };
		template<MathDomain md>
		static void SparseAdd(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha);
		template<>
		inline void SparseAdd<MathDomain::Float>(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
		{
			Copy<MathDomain::Float>(z, y);
			mkl::cblas_saxpyi(static_cast<int>(x.size), static_cast<float>(alpha), reinterpret_cast<float*>(x.pointer), reinterpret_cast<int*>(x.indices), reinterpret_cast<float*>(y.pointer));
		}
		template<>
		inline void SparseAdd<MathDomain::Double>(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
		{
			Copy<MathDomain::Double>(z, y);
			mkl::cblas_daxpyi(static_cast<int>(x.size), alpha, reinterpret_cast<double*>(x.pointer), reinterpret_cast<int*>(x.indices), reinterpret_cast<double*>(y.pointer));
		}

		template<MathDomain md>
		static void SparseDot(MemoryBuffer& y, const SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta);
		template<>
		inline void SparseDot<MathDomain::Float>(MemoryBuffer& y, const SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
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
			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_s_mv(mklSparseOperations[static_cast<size_t>(aOperation)],
								 static_cast<float>(alpha),
								 sparseA, descr,
								 reinterpret_cast<float*>(x.pointer),
								 static_cast<float>(beta), reinterpret_cast<float*>(y.pointer));

			mkl::mkl_sparse_destroy(sparseA);
		}
		template<>
		inline void SparseDot<MathDomain::Double>(MemoryBuffer& y, const SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
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
			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_d_mv(mklSparseOperations[static_cast<size_t>(aOperation)],
								 alpha,
								 sparseA, descr,
								 reinterpret_cast<double*>(x.pointer),
								 beta, reinterpret_cast<double*>(y.pointer));

			mkl::mkl_sparse_destroy(sparseA);
		}

		template<MathDomain md>
		static void SparseMultiply(MemoryTile& A, const SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha);
		template<>
		inline void SparseMultiply<MathDomain::Float>(MemoryTile& A, const SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
		{
			mkl::sparse_matrix_t sparseB;
			mkl::mkl_sparse_s_create_csr(&sparseB,
										 mkl::sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
										 static_cast<int>(B.nRows),
										 static_cast<int>(B.nCols),
										 reinterpret_cast<int*>(B.nNonZeroRows),
										 reinterpret_cast<int*>(B.nNonZeroRows) + 1,
										 reinterpret_cast<int*>(B.nonZeroColumnIndices),
										 reinterpret_cast<float*>(B.pointer));
			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_s_mm(mklSparseOperations[static_cast<size_t>(bOperation)],
								 static_cast<float>(alpha),
								 sparseB, descr,
								 mkl::sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,
								 reinterpret_cast<float*>(A.pointer), static_cast<int>(C.nCols), static_cast<int>(A.leadingDimension),
								 0.0f, reinterpret_cast<float*>(C.pointer), static_cast<int>(C.leadingDimension));

			mkl::mkl_sparse_destroy(sparseB);
		}
		template<>
		inline void SparseMultiply<MathDomain::Double>(MemoryTile& A, const SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
		{
			mkl::sparse_matrix_t sparseB;
			mkl::mkl_sparse_d_create_csr(&sparseB,
										 mkl::sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
										 static_cast<int>(B.nRows),
										 static_cast<int>(B.nCols),
										 reinterpret_cast<int*>(B.nNonZeroRows),
										 reinterpret_cast<int*>(B.nNonZeroRows) + 1,
										 reinterpret_cast<int*>(B.nonZeroColumnIndices),
										 reinterpret_cast<double*>(B.pointer));
			mkl::matrix_descr descr {};
			descr.diag = mkl::sparse_diag_type_t::SPARSE_DIAG_NON_UNIT;
			descr.mode = mkl::sparse_fill_mode_t::SPARSE_FILL_MODE_FULL;
			descr.type = mkl::sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL;
			mkl::mkl_sparse_d_mm(mklSparseOperations[static_cast<size_t>(bOperation)],
								 alpha,
								 sparseB, descr,
								 mkl::sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,
								 reinterpret_cast<double*>(A.pointer), static_cast<int>(C.nCols), static_cast<int>(A.leadingDimension),
								 0.0, reinterpret_cast<double*>(C.pointer), static_cast<int>(C.leadingDimension));

			mkl::mkl_sparse_destroy(sparseB);
		}

	}}}

#endif
