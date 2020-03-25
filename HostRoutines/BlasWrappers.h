#pragma once

#include <Types.h>

namespace cl { namespace routines {
	/**
	* z = alpha * x + y
	*/
	extern void Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha = 1.0);
	
	/**
	* z = x - y
	*/
	extern void Subtract(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y);
	
	/**
	* z += alpha * x
	*/
	extern void AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha = 1.0);
	
	/**
	* A = alpha * A + beta * B (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
	*/
	extern void AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation = MatrixOperation::None, const MatrixOperation bOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 1.0);
	
	/**
	* z -= x
	*/
	extern void SubtractEqual(MemoryBuffer& z, const MemoryBuffer& x);
	
	/**
	* z *= alpha
	*/
	extern void Scale(MemoryBuffer& z, const double alpha);
	
	/**
	* z[i, j] *= alpha[j]
	*/
	extern void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha);
	
	/**
	* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
	*/
	extern void ElementwiseProduct(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha = 1.0);
	
	/**
	* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
	*/
	extern void ElementwiseDivision(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha = 1.0);
	
	/*
	*	A = alpha * B * C + beta * A
	*/
	extern void Multiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	
	/*
	*	A = alpha * B * C + beta * A
	*/
	extern void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	
	/*
	*	A[i] = alpha * B[i] * C[i] + beta * A[i]
	*/
	extern void BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	
	/**
	*	y = alpha * A * x + beta * y
	*/
	extern void Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
	
	/**
	*	A += alpha * x * y^T
	*/
	extern void KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha = 1.0);
	
	/**
	*	T[i] += alpha * A[i] * B[i]^T,
	 *	NB: Instead of writing in A's depth, we're writing in A columns, so that effectively A is a collection of matrices.
	 *	    This helps when using NN gradient descent
	*/
	extern void BatchedTransposedKroneckerProduct(MemoryCube& T, const MemoryTile& x, const MemoryTile& y, const double alpha = 1.0);
	
	/**
	* A = cumsum(A)
	*/
	extern void CumulativeRowSum(MemoryTile& A);
	
	/**
	* x = sum(A[:, ])
	*/
	extern void RowWiseSum(MemoryBuffer& x, const MemoryTile& A, MemoryBuffer& cache, const MatrixOperation aOperation = MatrixOperation::None);
	
	/**
	* x = sum(A[:, ])
	*/
	extern void CubeWiseSum(MemoryTile& A, const MemoryCube& T, MemoryCube& cacheReshape, MemoryBuffer& cacheOnes);
	
	/**
	* X such that A * X = B by means of LU factorization
	*/
	extern void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation = MatrixOperation::None, const LinearSystemSolverType solver = LinearSystemSolverType::Lu);
	
	/**
	* A = A^(-1) by means of LU factorization
	*/
	extern void Invert(MemoryTile& A, const MatrixOperation aOperation = MatrixOperation::None);
	
	extern void ArgAbsMin(int& argMin, const MemoryBuffer& x);
	
	// NB: it returns 1-based indices
	extern void ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A);
	
	extern void ArgAbsMax(int& argMax, const MemoryBuffer& x);
	
	// NB: it returns 1-based indices
	extern void ColumnWiseArgAbsMax(MemoryBuffer& argMax, const MemoryTile& A);
	
	// z = { 1 if x == 0; 0 otherwise }
	extern void IsNonZero(MemoryBuffer& z, const MemoryBuffer& x);
	
	// norm = ||x||_2
	extern void EuclideanNorm(double& norm, const MemoryBuffer& x);
}}
