
#include <Common.h>
#include <Types.h>
#include <Exceptions.h>

#include <BlasWrappers.h>
#include <cmath>

namespace cl { namespace routines {
	/**
	* z = alpha * x + y
	*/
	void Add(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.memorySpace == y.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.mathDomain == y.mathDomain);
		assert(z.size == x.size);
		assert(z.size == y.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = static_cast<float>(alpha) * xPtr[i] + yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = alpha * xPtr[i] + yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = static_cast<int>(alpha) * xPtr[i] + yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	* z = x - y
	*/
	void Subtract(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.memorySpace == y.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.mathDomain == y.mathDomain);
		assert(z.size == x.size);
		assert(z.size == y.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] - yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] - yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] - yPtr[i];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	* z += alpha * x
	*/
	void AddEqual(MemoryBuffer& z, const MemoryBuffer& x, const double alpha)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.size == x.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * static_cast<float>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * alpha;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * static_cast<int>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

//	/**
//	* A = alpha * A + beta * B (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
//	*/
//	void AddEqualMatrix(MemoryTile&, const MemoryTile&, const MatrixOperation, const MatrixOperation, const double, const double)
//	{
//		throw NotImplementedException();
//	}

	/**
	* z -= x
	*/
	void SubtractEqual(MemoryBuffer& z, const MemoryBuffer& x)
	{
		AddEqual(z, x, -1.0);
	}

	/**
	* z *= alpha
	*/
	void Scale(MemoryBuffer& z, const double alpha)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= static_cast<float>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= alpha;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= static_cast<int>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	* z[i, j] *= alpha[j]
	*/
	void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *aPtr = GetPointer<MathDomain::Float>(alpha);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *aPtr = GetPointer<MathDomain::Double>(alpha);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *aPtr = GetPointer<MathDomain::Int>(alpha);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
	*/
	void ElementwiseProduct(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.memorySpace == y.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.mathDomain == y.mathDomain);
		assert(z.size == x.size);
		assert(z.size == y.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * static_cast<float>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * alpha;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * static_cast<int>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	* z = alpha * x * y: NB: there's no such a function in cuBLAS -> I use SBMV with a diagonal matrix == vector
	*/
	void ElementwiseDivision(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.memorySpace == y.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.mathDomain == y.mathDomain);
		assert(z.size == x.size);
		assert(z.size == y.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * static_cast<float>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * alpha;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * static_cast<int>(alpha);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

//	/*
//	*	A = alpha * B * C + beta * A
//	*/
//	void Multiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
//
//	/*
//	*	A = alpha * B * C + beta * A
//	*/
//	void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);
//
//	/*
//	*	A[i] = alpha * B[i] * C[i] + beta * A[i]
//	*/
//	void BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation = MatrixOperation::None, const MatrixOperation cOperation = MatrixOperation::None, const double alpha = 1.0, const double beta = 0.0);

	/**
	*	y = alpha * A * x + beta * y
	*/
	void Dot(MemoryBuffer& y, const MemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
	{
		assert(y.memorySpace == A.memorySpace);
		assert(y.memorySpace == x.memorySpace);
		assert(y.mathDomain == A.mathDomain);
		assert(y.mathDomain == x.mathDomain);

		switch (y.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *aPtr = GetPointer<MathDomain::Float>(x);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
						{
							yPtr[i] = yPtr[i] * static_cast<float>(beta);
							for (size_t j = 0; j < A.nCols; ++j)
							{
								if (aOperation == MatrixOperation::None)
									yPtr[i] += aPtr[i + j * A.nRows] * xPtr[j] * static_cast<float>(alpha);
								else if (aOperation == MatrixOperation::Transpose)
									yPtr[i] += aPtr[j + i * A.nCols] * xPtr[j] * static_cast<float>(alpha);
							}
						}
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *aPtr = GetPointer<MathDomain::Double>(x);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
						{
							yPtr[i] = yPtr[i] * beta;
							for (size_t j = 0; j < A.nCols; ++j)
								yPtr[i] += aPtr[i + j * A.nRows] * xPtr[j] * alpha;
						}
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *aPtr = GetPointer<MathDomain::Int>(x);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
						{
							yPtr[i] = yPtr[i] * static_cast<int>(beta);
							for (size_t j = 0; j < A.nCols; ++j)
								yPtr[i] += aPtr[i + j * A.nRows] * xPtr[j] * static_cast<int>(alpha);
						}
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	/**
	*	A += alpha * x * y^T
	*/
	void KroneckerProduct(MemoryTile& A, const MemoryBuffer& x, const MemoryBuffer& y, const double alpha)
	{
		assert(y.memorySpace == A.memorySpace);
		assert(y.memorySpace == x.memorySpace);
		assert(y.mathDomain == A.mathDomain);
		assert(y.mathDomain == x.mathDomain);

		switch (y.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *aPtr = GetPointer<MathDomain::Float>(x);
				auto *xPtr = GetPointer<MathDomain::Float>(x);
				auto *yPtr = GetPointer<MathDomain::Float>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += static_cast<float>(alpha) * xPtr[i] * yPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *aPtr = GetPointer<MathDomain::Double>(x);
				auto *xPtr = GetPointer<MathDomain::Double>(x);
				auto *yPtr = GetPointer<MathDomain::Double>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += alpha * xPtr[i] * yPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *aPtr = GetPointer<MathDomain::Int>(x);
				auto *xPtr = GetPointer<MathDomain::Int>(x);
				auto *yPtr = GetPointer<MathDomain::Int>(y);

				switch (y.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += static_cast<int>(alpha) * xPtr[i] * yPtr[j];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

//	/**
//	*	T[i] += alpha * A[i] * B[i]^T,
//	 *	NB: Instead of writing in A's depth, we're writing in A columns, so that effectively A is a collection of matrices.
//	 *	    This helps when using NN gradient descent
//	*/
//	void BatchedTransposedKroneckerProduct(MemoryCube& T, const MemoryTile& x, const MemoryTile& y, const double alpha = 1.0);

	/**
	* A = cumsum(A)
	*/
	void CumulativeRowSum(MemoryTile& A)
	{
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *aPtr = GetPointer<MathDomain::Float>(A);

				switch (A.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
							for (size_t j = 1; j < A.nCols; ++j)
								aPtr[i + j * A.nRows] += aPtr[i + (j - 1) * A.nRows];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *aPtr = GetPointer<MathDomain::Double>(A);

				switch (A.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
							for (size_t j = 1; j < A.nCols; ++j)
								aPtr[i + j * A.nRows] += aPtr[i + (j - 1) * A.nRows];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *aPtr = GetPointer<MathDomain::Int>(A);

				switch (A.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < A.nRows; ++i)
							for (size_t j = 1; j < A.nCols; ++j)
								aPtr[i + j * A.nRows] += aPtr[i + (j - 1) * A.nRows];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

//	/**
//	* x = sum(A[:, ])
//	*/
//	void RowWiseSum(MemoryBuffer& x, const MemoryTile& A, MemoryBuffer& cache, const MatrixOperation aOperation = MatrixOperation::None)
//	{
//
//	}
//
//	/**
//	* x = sum(A[:, ])
//	*/
//	void CubeWiseSum(MemoryTile& A, const MemoryCube& T, MemoryCube& cacheReshape, MemoryBuffer& cacheOnes)
//	{
//
//	}

//	/**
//	* X such that A * X = B by means of LU factorization
//	*/
//	void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation = MatrixOperation::None)
//	{
//
//	}
//
//	/**
//	* A = A^(-1) by means of LU factorization
//	*/
//	void Invert(MemoryTile& A, const MatrixOperation aOperation = MatrixOperation::None)
//	{
//
//	}

	void ArgAbsMin(int& argMin, const MemoryBuffer& x)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto min = std::fabs(xPtr[0]);
						argMin = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) < min)
							{
								min = std::fabs(xPtr[i]);
								argMin = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto min = std::fabs(xPtr[0]);
						argMin = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) < min)
							{
								min = std::fabs(xPtr[i]);
								argMin = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto min = std::fabs(xPtr[0]);
						argMin = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) < min)
							{
								min = std::fabs(xPtr[i]);
								argMin = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	// NB: it returns 1-based indices
	void ColumnWiseArgAbsMin(MemoryBuffer& argMin, const MemoryTile& A)
	{
		assert(A.nCols == argMin.size);
		assert(A.memorySpace == argMin.memorySpace);
		assert(argMin.mathDomain == MathDomain::Int);

		auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMin(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMin(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMin(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	void ArgAbsMax(int& argMax, const MemoryBuffer& x)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto max = std::fabs(xPtr[0]);
						argMax = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) > max)
							{
								max = std::fabs(xPtr[i]);
								argMax = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto max = std::fabs(xPtr[0]);
						argMax = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) > max)
							{
								max = std::fabs(xPtr[i]);
								argMax = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto max = std::fabs(xPtr[0]);
						argMax = 0;
						for (size_t i = 1; i < x.size; ++i)
						{
							if (std::fabs(xPtr[i]) > max)
							{
								max = std::fabs(xPtr[i]);
								argMax = static_cast<int>(i);
							}
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	// NB: it returns 1-based indices
	void ColumnWiseArgAbsMax(MemoryBuffer& argMin, const MemoryTile& A)
	{
		assert(A.nCols == argMin.size);
		assert(A.memorySpace == argMin.memorySpace);
		assert(argMin.mathDomain == MathDomain::Int);

		auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMax(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMax(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Test:
					{
						MemoryBuffer tmp;
						for (size_t j = 0; j < A.nCols; ++j)
						{
							ExtractColumnBufferFromMatrix(tmp, A, static_cast<unsigned>(j));
							ArgAbsMax(argMinPtr[j], tmp);
							++argMinPtr[j];  //NB: for compatibility we have to return 1-based indices
						}
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	// z = { 1 if x == 0; 0 otherwise }
	void IsNonZero(MemoryBuffer& z, const MemoryBuffer& x)
	{
		assert(z.memorySpace == x.memorySpace);
		assert(z.mathDomain == x.mathDomain);
		assert(z.size == x.size);

		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *zPtr = GetPointer<MathDomain::Float>(z);
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = std::fabs(xPtr[i]) < 1e-7f ? 0 : 1;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *zPtr = GetPointer<MathDomain::Double>(z);
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = std::fabs(xPtr[i]) < 1e-7 ? 0 : 1;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *zPtr = GetPointer<MathDomain::Int>(z);
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (z.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] == 0 ? 0 : 1;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}

	// norm = ||x||_2
	void EuclideanNorm(double& norm, const MemoryBuffer& x)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
						norm = 0.0;
						for (size_t i = 0; i < x.size; ++i)
							norm += static_cast<double>(xPtr[i] * xPtr[i]);
						norm = std::sqrt(norm);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
						norm = 0.0;
						for (size_t i = 0; i < x.size; ++i)
							norm += xPtr[i] * xPtr[i];
						norm = std::sqrt(norm);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
						norm = 0.0;
						for (size_t i = 0; i < x.size; ++i)
							norm += static_cast<double>(xPtr[i] * xPtr[i]);
						norm = std::sqrt(norm);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			default:
				throw NotImplementedException();
		}
	}
}}