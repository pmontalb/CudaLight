#pragma once

#include <Types.h>

namespace cl { namespace routines {
	extern void AllocateCsrHandle(SparseMemoryTile& A);

	extern void DestroyCsrHandle(SparseMemoryTile& A);

	/**
	* zDense = alpha * xSparse + yDense
	*/
	extern void SparseAdd(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha = 1.0);

	/**
	* zDense = yDense - xSparse
	*/
	extern void SparseSubtract(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y);

	/**
	*	yDense = ASparse * xDense
	*/
	extern void SparseDot(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);

	/**
	*	ADense = BSparse * CDense
	*/
	extern void SparseMultiply(MemoryTile& A, SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0);

	extern void SparseSolve(SparseMemoryTile& A, MemoryTile& B, LinearSystemSolverType solver = LinearSystemSolverType::Lu);
}}
