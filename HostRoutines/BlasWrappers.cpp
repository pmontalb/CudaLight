
#include <Common.h>
#include <Types.h>
#include <Exceptions.h>
#include <MemoryManager.h>

#include <BlasWrappers.h>
#include <MklAllWrappers.h>
#include <OpenBlasAllWrappers.h>
#include <GenericBlasAllWrappers.h>

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
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Add<MathDomain::Float>(z, x, y, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::Add<MathDomain::Float>(z, x, y, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::Add<MathDomain::Float>(z, x, y, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *xPtr = GetPointer<MathDomain::Float>(x);
						auto *yPtr = GetPointer<MathDomain::Float>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = static_cast<float>(alpha) * xPtr[i] + yPtr[i];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Add<MathDomain::Double>(z, x, y, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::Add<MathDomain::Double>(z, x, y, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::Add<MathDomain::Double>(z, x, y, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *xPtr = GetPointer<MathDomain::Double>(x);
						auto *yPtr = GetPointer<MathDomain::Double>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = alpha * xPtr[i] + yPtr[i];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl: // TODO
					case MemorySpace::OpenBlas: // TODO
					case MemorySpace::GenericBlas: // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						auto *yPtr = GetPointer<MathDomain::Int>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = static_cast<int>(alpha) * xPtr[i] + yPtr[i];
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

	/**
	* z = x - y
	*/
	void Subtract(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y)
	{
		Add(z, x, y, -1.0);
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
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::AddEqual<MathDomain::Float>(z, x, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::AddEqual<MathDomain::Float>(z, x, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::AddEqual<MathDomain::Float>(z, x, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *xPtr = GetPointer<MathDomain::Float>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * static_cast<float>(alpha);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::AddEqual<MathDomain::Double>(z, x, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::AddEqual<MathDomain::Double>(z, x, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::AddEqual<MathDomain::Double>(z, x, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *xPtr = GetPointer<MathDomain::Double>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * alpha;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *xPtr = GetPointer<MathDomain::Int>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] += xPtr[i] * static_cast<int>(alpha);
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

	/**
	* A = alpha * A + beta * B (NB: it uses <t>geam, maybe more efficient than <t>axpy?)
	*/
	void AddEqualMatrix(MemoryTile& A, const MemoryTile& B, const MatrixOperation aOperation, const MatrixOperation bOperation, const double alpha, const double beta)
	{
		assert(A.memorySpace == B.memorySpace);
		assert(B.mathDomain == B.mathDomain);

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::AddEqualMatrix<MathDomain::Float>(A, B, aOperation, bOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::AddEqualMatrix<MathDomain::Float>(A, B, aOperation, bOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::AddEqualMatrix<MathDomain::Float>(A, B, aOperation, bOperation, alpha, beta);
						break;

					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::AddEqualMatrix<MathDomain::Double>(A, B, aOperation, bOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::AddEqualMatrix<MathDomain::Double>(A, B, aOperation, bOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::AddEqualMatrix<MathDomain::Double>(A, B, aOperation, bOperation, alpha, beta);
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
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Scale<MathDomain::Float>(z, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::Scale<MathDomain::Float>(z, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::Scale<MathDomain::Float>(z, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= static_cast<float>(alpha);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Scale<MathDomain::Double>(z, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::Scale<MathDomain::Double>(z, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::Scale<MathDomain::Double>(z, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= alpha;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] *= static_cast<int>(alpha);
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

	/**
	* z[i, j] *= alpha[j]
	*/
	void ScaleColumns(MemoryTile& z, const MemoryBuffer& alpha)
	{
		switch (z.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ScaleColumns<MathDomain::Float>(z, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::ScaleColumns<MathDomain::Float>(z, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::ScaleColumns<MathDomain::Float>(z, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *aPtr = GetPointer<MathDomain::Float>(alpha);

						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ScaleColumns<MathDomain::Double>(z, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::ScaleColumns<MathDomain::Double>(z, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::ScaleColumns<MathDomain::Double>(z, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *aPtr = GetPointer<MathDomain::Double>(alpha);

						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *aPtr = GetPointer<MathDomain::Int>(alpha);

						for (size_t j = 0; j < z.nCols; ++j)
							for (size_t i = 0; i < z.nRows; ++i)
								zPtr[i + j * z.nRows] *= aPtr[j];
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
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ElementwiseProduct<MathDomain::Float>(z, x, y, alpha);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *xPtr = GetPointer<MathDomain::Float>(x);
						auto *yPtr = GetPointer<MathDomain::Float>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * static_cast<float>(alpha);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ElementwiseProduct<MathDomain::Double>(z, x, y, alpha);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *xPtr = GetPointer<MathDomain::Double>(x);
						auto *yPtr = GetPointer<MathDomain::Double>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * alpha;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						auto *yPtr = GetPointer<MathDomain::Int>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] * yPtr[i] * static_cast<int>(alpha);
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
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *xPtr = GetPointer<MathDomain::Float>(x);
						auto *yPtr = GetPointer<MathDomain::Float>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * static_cast<float>(alpha);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *xPtr = GetPointer<MathDomain::Double>(x);
						auto *yPtr = GetPointer<MathDomain::Double>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * alpha;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						auto *yPtr = GetPointer<MathDomain::Int>(y);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] / yPtr[i] * static_cast<int>(alpha);
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

	/*
	*	A = alpha * B * C + beta * A
	*/
	void Multiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		SubMultiply(A, B, C, B.nRows, B.nCols, C.nCols, bOperation, cOperation, alpha, beta);
	}

	/*
	*	A = alpha * B * C + beta * A
	*/
	void SubMultiply(MemoryTile& A, const MemoryTile& B, const MemoryTile& C, const unsigned nRowsB, const unsigned nColsB, const unsigned nColsC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		assert(A.memorySpace == B.memorySpace);
		assert(A.memorySpace == C.memorySpace);
		assert(A.mathDomain == A.mathDomain);
		assert(A.mathDomain == C.mathDomain);

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::SubMultiply<MathDomain::Float>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::SubMultiply<MathDomain::Float>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::SubMultiply<MathDomain::Float>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Float>(A);
						auto *bPtr = GetPointer<MathDomain::Float>(B);
						auto *cPtr = GetPointer<MathDomain::Float>(C);
						const auto _alpha = static_cast<float>(alpha);
						const auto _beta = static_cast<float>(beta);

						// TODO transpositions!
						for(size_t i = 0; i < nRowsB; ++i)
							for (size_t k = 0; k < nColsC; ++k)
								for (size_t j = 0; j < nColsB; ++j)
									aPtr[i + k * A.leadingDimension] = _beta * aPtr[i + k * A.leadingDimension] + _alpha * bPtr[i + j * B.leadingDimension] * cPtr[j + k * C.leadingDimension];
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
					case MemorySpace::Mkl:
						mkr::SubMultiply<MathDomain::Double>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::SubMultiply<MathDomain::Double>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::SubMultiply<MathDomain::Double>(A, B, C, nRowsB, nColsB, nColsC, bOperation, cOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Double>(A);
						auto *bPtr = GetPointer<MathDomain::Double>(B);
						auto *cPtr = GetPointer<MathDomain::Double>(C);

						// TODO transpositions!
						for(size_t i = 0; i < B.nRows; ++i)
							for (size_t k = 0; k < C.nCols; ++k)
								for (size_t j = 0; j < B.nCols; ++j)
									aPtr[i + k * A.leadingDimension] = beta * aPtr[i + k * A.leadingDimension] + alpha * bPtr[i + j * B.leadingDimension] * cPtr[j + k * C.leadingDimension];
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
					case MemorySpace::Mkl: // TODO
					case MemorySpace::OpenBlas: // TODO
					case MemorySpace::GenericBlas: // TODO
					{
						auto *aPtr = GetPointer<MathDomain::Int>(A);
						auto *bPtr = GetPointer<MathDomain::Int>(B);
						auto *cPtr = GetPointer<MathDomain::Int>(C);

						const auto _alpha = static_cast<int>(alpha);
						const auto _beta = static_cast<int>(beta);

						// TODO transpositions!
						for(size_t i = 0; i < B.nRows; ++i)
							for (size_t k = 0; k < C.nCols; ++k)
								for (size_t j = 0; j < B.nCols; ++j)
									aPtr[i + k * A.leadingDimension] = _beta * aPtr[i + k * A.leadingDimension] + _alpha * bPtr[i + j * B.leadingDimension] * cPtr[j + k * C.leadingDimension];
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

	/*
	*	A[i] = alpha * B[i] * C[i] + beta * A[i]
	*/
	void BatchedMultiply(MemoryCube& A, const MemoryCube& B, const MemoryCube& C, const unsigned strideB, const unsigned strideC, const MatrixOperation bOperation, const MatrixOperation cOperation, const double alpha, const double beta)
	{
		assert(A.memorySpace == B.memorySpace);
		assert(A.memorySpace == C.memorySpace);
		assert(A.mathDomain == A.mathDomain);
		assert(A.mathDomain == C.mathDomain);
		assert(A.nCubes == B.nCubes);
		assert(A.nCubes == C.nCubes);

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::BatchedMultiply<MathDomain::Float>(A, B, C, strideB, strideC, bOperation, cOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						for (unsigned n = 0; n < A.nCubes; ++n)
						{
							MemoryTile a, b, c;
							ExtractMatrixBufferFromCube(a, A, n);
							ExtractMatrixBufferFromCube(b, B, n);
							ExtractMatrixBufferFromCube(c, C, n);
							Multiply(A, B, C, bOperation, cOperation, alpha, beta);
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
					case MemorySpace::Mkl:
						mkr::BatchedMultiply<MathDomain::Double>(A, B, C, strideB, strideC, bOperation, cOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						for (unsigned n = 0; n < A.nCubes; ++n)
						{
							MemoryTile a, b, c;
							ExtractMatrixBufferFromCube(a, A, n);
							ExtractMatrixBufferFromCube(b, B, n);
							ExtractMatrixBufferFromCube(c, C, n);
							Multiply(A, B, C, bOperation, cOperation, alpha, beta);
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
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						for (unsigned n = 0; n < A.nCubes; ++n)
						{
							MemoryTile a, b, c;
							ExtractMatrixBufferFromCube(a, A, n);
							ExtractMatrixBufferFromCube(b, B, n);
							ExtractMatrixBufferFromCube(c, C, n);
							Multiply(A, B, C, bOperation, cOperation, alpha, beta);
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
				switch (y.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Dot<MathDomain::Float>(y, A, x, aOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::Dot<MathDomain::Float>(y, A, x, aOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::Dot<MathDomain::Float>(y, A, x, aOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Float>(x);
						auto *xPtr = GetPointer<MathDomain::Float>(x);
						auto *yPtr = GetPointer<MathDomain::Float>(y);

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
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (y.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Dot<MathDomain::Double>(y, A, x, aOperation, alpha, beta);
						break;
					case MemorySpace::OpenBlas:
						obr::Dot<MathDomain::Double>(y, A, x, aOperation, alpha, beta);
						break;
					case MemorySpace::GenericBlas:
						gbr::Dot<MathDomain::Double>(y, A, x, aOperation, alpha, beta);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Double>(x);
						auto *xPtr = GetPointer<MathDomain::Double>(x);
						auto *yPtr = GetPointer<MathDomain::Double>(y);

						for (size_t i = 0; i < A.nRows; ++i)
						{
							yPtr[i] = yPtr[i] * beta;
							for (size_t j = 0; j < A.nCols; ++j)
							{
								if (aOperation == MatrixOperation::None)
									yPtr[i] += aPtr[i + j * A.nRows] * xPtr[j] * alpha;
								else if (aOperation == MatrixOperation::Transpose)
									yPtr[i] += aPtr[j + i * A.nCols] * xPtr[j] * alpha;
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
				switch (y.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *aPtr = GetPointer<MathDomain::Int>(x);
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						auto *yPtr = GetPointer<MathDomain::Int>(y);

						for (size_t i = 0; i < A.nRows; ++i)
						{
							yPtr[i] = yPtr[i] * static_cast<int>(beta);
							for (size_t j = 0; j < A.nCols; ++j)
								yPtr[i] += aPtr[i + j * A.nRows] * xPtr[j] * static_cast<int>(alpha);
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
				switch (y.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::KroneckerProduct<MathDomain::Float>(A, x, y, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::KroneckerProduct<MathDomain::Float>(A, x, y, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::KroneckerProduct<MathDomain::Float>(A, x, y, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Float>(x);
						auto *xPtr = GetPointer<MathDomain::Float>(x);
						auto *yPtr = GetPointer<MathDomain::Float>(y);

						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += static_cast<float>(alpha) * xPtr[i] * yPtr[j];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (y.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::KroneckerProduct<MathDomain::Double>(A, x, y, alpha);
						break;
					case MemorySpace::OpenBlas:
						obr::KroneckerProduct<MathDomain::Double>(A, x, y, alpha);
						break;
					case MemorySpace::GenericBlas:
						gbr::KroneckerProduct<MathDomain::Double>(A, x, y, alpha);
						break;

					case MemorySpace::Test:
					{
						auto *aPtr = GetPointer<MathDomain::Double>(x);
						auto *xPtr = GetPointer<MathDomain::Double>(x);
						auto *yPtr = GetPointer<MathDomain::Double>(y);

						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += alpha * xPtr[i] * yPtr[j];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (y.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl: // TODO
					case MemorySpace::OpenBlas: // TODO
					case MemorySpace::GenericBlas: // TODO
					{
						auto *aPtr = GetPointer<MathDomain::Int>(x);
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						auto *yPtr = GetPointer<MathDomain::Int>(y);

						for (size_t j = 0; j < A.nCols; ++j)
							for (size_t i = 0; i < A.nRows; ++i)
								aPtr[i + j * A.nRows] += static_cast<int>(alpha) * xPtr[i] * yPtr[j];
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

	/**
	*	T[i] += alpha * A[i] * B[i]^T,
	 *	NB: Instead of writing in A's depth, we're writing in A columns, so that effectively A is a collection of matrices.
	 *	    This helps when using NN gradient descent
	*/
	void BatchedTransposedKroneckerProduct(MemoryCube& T, const MemoryTile& x, const MemoryTile& y, const double alpha)
	{
		assert(T.memorySpace == x.memorySpace);
		assert(T.memorySpace == y.memorySpace);
		assert(T.mathDomain == x.mathDomain);
		assert(T.mathDomain == y.mathDomain);
		assert(T.nCubes == x.nCols);
		assert(T.nCubes == y.nCols);

		switch (T.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (T.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						MemoryTile t {};
						MemoryBuffer _x {}, _y {};

						for (unsigned n = 0; n < T.nCubes; ++n)
						{
							ExtractMatrixBufferFromCube(t, T, n);
							ExtractColumnBufferFromMatrix(_x, x, n);
							ExtractColumnBufferFromMatrix(_y, y, n);
							KroneckerProduct(t, _x, _y, alpha);
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
				switch (T.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						MemoryTile t {};
						MemoryBuffer _x {}, _y {};

						for (unsigned n = 0; n < T.nCubes; ++n)
						{
							ExtractMatrixBufferFromCube(t, T, n);
							ExtractColumnBufferFromMatrix(_x, x, n);
							ExtractColumnBufferFromMatrix(_y, y, n);
							KroneckerProduct(t, _x, _y, alpha);
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
				switch (T.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						MemoryTile t {};
						MemoryBuffer _x {}, _y {};

						for (unsigned n = 0; n < T.nCubes; ++n)
						{
							ExtractMatrixBufferFromCube(t, T, n);
							ExtractColumnBufferFromMatrix(_x, x, n);
							ExtractColumnBufferFromMatrix(_y, y, n);
							KroneckerProduct(t, _x, _y, alpha);
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

	/**
	* A = cumsum(A)
	*/
	void CumulativeRowSum(MemoryTile& A)
	{
		MemoryTile ones(0, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
		Alloc(ones);
		OnesUpperTriangular(ones);

		MemoryTile buffer(0, A.nRows, A.nCols, A.memorySpace, A.mathDomain);
		Alloc(buffer);

		Copy(buffer, A);

		Multiply(A, ones, buffer);

		Free(ones);
		Free(buffer);
	}

	/**
	* x = sum(A[:, ])
	*/
	void RowWiseSum(MemoryBuffer& x, const MemoryTile& A, MemoryBuffer& cache, const MatrixOperation aOperation)
	{
		auto dim = aOperation == MatrixOperation::None ? A.nCols : A.nRows;
		if (cache.size != dim)
		{
			if (cache.pointer != 0)
				Free(cache);
			cache.pointer = 0;
		}

		if (cache.pointer == 0)
		{
			cache = MemoryBuffer(cache.pointer, dim, A.memorySpace, A.mathDomain);
			Alloc(cache);
			Initialize(cache, 1.0);
		}

		Dot(x, A, cache, aOperation);
	}

	/**
	* x = sum(A[:, ])
	*/
	void CubeWiseSum(MemoryTile& A, const MemoryCube& T, MemoryCube& cacheReshape, MemoryBuffer& cacheOnes)
	{
		if (cacheOnes.size != T.nCubes)
		{
			if (cacheOnes.pointer != 0)
				Free(cacheOnes);
			cacheOnes.pointer = 0;
		}

		if (cacheOnes.pointer == 0)
		{
			cacheOnes = MemoryBuffer(0, T.nCubes, T.memorySpace, T.mathDomain);
			Alloc(cacheOnes);
			Initialize(cacheOnes, 1.0);
		}

		// reshape T into nCols blocks of [nRows * nCubes]
		if (cacheReshape.nRows != T.nRows || cacheReshape.nCols != T.nCubes || cacheReshape.nCubes != T.nCols)
		{
			if (cacheReshape.pointer != 0)
				Free(cacheReshape);
			cacheReshape.pointer = 0;
		}

		if (cacheReshape.pointer == 0)
		{
			cacheReshape = MemoryCube(0, T.nRows, T.nCubes, T.nCols, T.memorySpace, T.mathDomain);
			Alloc(cacheReshape);
		}

		auto reshapeWorker = [](auto* RESTRICT out, const auto* RESTRICT in, const size_t nRows, const size_t nCols, const size_t nCubes)
		{
			const size_t inMatrixSize = nRows * nCols;
			const size_t outMatrixSize = nRows * nCubes;
			for (size_t i = 0; i < nRows; ++i)
			{
				for (size_t j = 0; j < nCols; ++j)
				{
					const size_t inStride = i + j * nRows;
					const size_t outOffset = i + j * outMatrixSize;
					for (size_t k = 0; k < nCubes; ++k)
						out[outOffset + k * nRows] = in[inStride + k * inMatrixSize];
				}
			}
		};

		switch (T.mathDomain)
		{
			case MathDomain::Float:
				reshapeWorker(reinterpret_cast<float *>(cacheReshape.pointer), reinterpret_cast<float *>(T.pointer), T.nRows, T.nCols, T.nCubes);
				break;
			case MathDomain::Double:
				reshapeWorker(reinterpret_cast<double *>(cacheReshape.pointer), reinterpret_cast<double *>(T.pointer), T.nRows, T.nCols, T.nCubes);
				break;
			case MathDomain::Int:
				reshapeWorker(reinterpret_cast<int *>(cacheReshape.pointer), reinterpret_cast<int *>(T.pointer), T.nRows, T.nCols, T.nCubes);
				break;
			default:
				throw NotImplementedException();
		}

		MemoryCube tmp1(A.pointer, A.nRows, 1, T.nCubes, A.memorySpace, A.mathDomain);
		MemoryCube tmp2(cacheReshape.pointer, cacheReshape.nRows, cacheReshape.nCols, 0, A.memorySpace, A.mathDomain);
		MemoryCube tmp3(cacheOnes.pointer, cacheOnes.size, 0, 0, A.memorySpace, A.mathDomain);
		BatchedMultiply(tmp1, tmp2, tmp3,cacheReshape.nRows * cacheReshape.nCols, 0);
	}

	/**
	* X such that A * X = B by means of LU factorization
	*/
	void Solve(const MemoryTile& A, MemoryTile& B, const MatrixOperation aOperation, const LinearSystemSolverType solver)
	{
		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Solve<MathDomain::Float>(A, B, aOperation, solver);
						break;
					case MemorySpace::OpenBlas:
						obr::Solve<MathDomain::Float>(A, B, aOperation, solver);
						break;
					case MemorySpace::GenericBlas:
						gbr::Solve<MathDomain::Float>(A, B, aOperation, solver);
						break;

					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Solve<MathDomain::Double>(A, B, aOperation, solver);
						break;
					case MemorySpace::OpenBlas:
						obr::Solve<MathDomain::Double>(A, B, aOperation, solver);
						break;
					case MemorySpace::GenericBlas:
						gbr::Solve<MathDomain::Double>(A, B, aOperation, solver);
						break;

					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			default:
				throw NotImplementedException();
		}
	}

	/**
	* A = A^(-1) by means of LU factorization
	*/
	void Invert(MemoryTile& A, const MatrixOperation aOperation)
	{
		MemoryTile eye(0, A.nRows, A.nRows, A.memorySpace, A.mathDomain);
		Alloc(eye);
		Eye(eye);

		// A^{-1} -> eye
		Solve(A, eye, aOperation);

		// eye -> A
		Copy(A, eye);

		Free(eye);
	}

	void ArgAbsMin(int& argMin, const MemoryBuffer& x)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (x.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ArgAbsMin<MathDomain::Float>(argMin, x);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas: // TODO: there's no isamin??
					case MemorySpace::GenericBlas: // TODO: there's no isamin??
					{
						auto *xPtr = GetPointer<MathDomain::Float>(x);

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
				switch (x.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ArgAbsMin<MathDomain::Double>(argMin, x);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas: // TODO: there's no isamin??
					case MemorySpace::GenericBlas: // TODO: there's no isamin??
					{
						auto *xPtr = GetPointer<MathDomain::Double>(x);

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
				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas: // TODO
					{
						auto *xPtr = GetPointer<MathDomain::Int>(x);

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

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ColumnWiseArgAbsMin<MathDomain::Float>(argMin, A);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas: // TODO: there's no isamin??
					case MemorySpace::GenericBlas: // TODO: there's no isamin??
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
					case MemorySpace::Mkl:
						mkr::ColumnWiseArgAbsMin<MathDomain::Double>(argMin, A);
						break;

					case MemorySpace::Test:
					case MemorySpace::OpenBlas: // TODO: there's no isamin??
					case MemorySpace::GenericBlas: // TODO: there's no isamin??
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas: // TODO
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
				switch (x.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ArgAbsMax<MathDomain::Float>(argMax, x);
						break;
					case MemorySpace::OpenBlas:
						obr::ArgAbsMax<MathDomain::Float>(argMax, x);
						break;
					case MemorySpace::GenericBlas:
						gbr::ArgAbsMax<MathDomain::Float>(argMax, x);
						break;

					case MemorySpace::Test:
					{
						auto *xPtr = GetPointer<MathDomain::Float>(x);

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
				switch (x.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ArgAbsMax<MathDomain::Double>(argMax, x);
						break;
					case MemorySpace::OpenBlas:
						obr::ArgAbsMax<MathDomain::Double>(argMax, x);
						break;
					case MemorySpace::GenericBlas:
						gbr::ArgAbsMax<MathDomain::Double>(argMax, x);
						break;

					case MemorySpace::Test:
					{
						auto *xPtr = GetPointer<MathDomain::Double>(x);

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
				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *xPtr = GetPointer<MathDomain::Int>(x);

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

		switch (A.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (A.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::ColumnWiseArgAbsMax<MathDomain::Float>(argMin, A);
						break;
					case MemorySpace::OpenBlas:
						obr::ColumnWiseArgAbsMax<MathDomain::Float>(argMin, A);
						break;
					case MemorySpace::GenericBlas:
						gbr::ColumnWiseArgAbsMax<MathDomain::Float>(argMin, A);
						break;

					case MemorySpace::Test:
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
					case MemorySpace::Mkl:
						mkr::ColumnWiseArgAbsMax<MathDomain::Double>(argMin, A);
						break;
					case MemorySpace::OpenBlas:
						obr::ColumnWiseArgAbsMax<MathDomain::Double>(argMin, A);
						break;
					case MemorySpace::GenericBlas:
						gbr::ColumnWiseArgAbsMax<MathDomain::Double>(argMin, A);
						break;

					case MemorySpace::Test:
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *argMinPtr = GetPointer<MathDomain::Int>(argMin);

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
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Float>(z);
						auto *xPtr = GetPointer<MathDomain::Float>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = std::fabs(xPtr[i]) < 1e-7f ? 0.0f : 1.0f;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Double>(z);
						auto *xPtr = GetPointer<MathDomain::Double>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = std::fabs(xPtr[i]) < 1e-7 ? 0 : 1;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (z.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *zPtr = GetPointer<MathDomain::Int>(z);
						auto *xPtr = GetPointer<MathDomain::Int>(x);

						for (size_t i = 0; i < z.size; ++i)
							zPtr[i] = xPtr[i] == 0 ? 0 : 1;
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

	// norm = ||x||_2
	void EuclideanNorm(double& norm, const MemoryBuffer& x)
	{
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (x.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::EuclideanNorm<MathDomain::Float>(norm, x);
						break;
					case MemorySpace::OpenBlas:
						obr::EuclideanNorm<MathDomain::Float>(norm, x);
						break;
					case MemorySpace::GenericBlas:
						gbr::EuclideanNorm<MathDomain::Float>(norm, x);
						break;

					case MemorySpace::Test:
					{
						auto *xPtr = GetPointer<MathDomain::Float>(x);

						norm = 0.0;
						for (size_t i = 0; i < x.size; ++i)
							norm += static_cast<double>(xPtr[i] * xPtr[i]);
						norm = std::sqrt(norm);
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
					case MemorySpace::Mkl:
						mkr::EuclideanNorm<MathDomain::Double>(norm, x);
						break;
					case MemorySpace::OpenBlas:
						obr::EuclideanNorm<MathDomain::Double>(norm, x);
						break;
					case MemorySpace::GenericBlas:
						gbr::EuclideanNorm<MathDomain::Double>(norm, x);
						break;

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
				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					case MemorySpace::OpenBlas:  // TODO
					case MemorySpace::GenericBlas:  // TODO
					{
						auto *xPtr = GetPointer<MathDomain::Int>(x);
						norm = 0.0;
						for (size_t i = 0; i < x.size; ++i)
							norm += static_cast<double>(xPtr[i] * xPtr[i]);
						norm = std::sqrt(norm);
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
}}
