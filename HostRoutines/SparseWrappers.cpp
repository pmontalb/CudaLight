
#include "Common.h"
#include <MklAllWrappers.h>
#include <SparseWrappers.h>

namespace cl
{
	namespace routines
	{
		void AllocateCsrHandle(SparseMemoryTile& A)
		{
			switch (A.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::AllocateCsrHandle<MathDomain::Float>(A);
							break;
						case MemorySpace::Test:
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
							mkr::AllocateCsrHandle<MathDomain::Double>(A);
							break;
						case MemorySpace::Test:
							break;
						default:
							throw NotImplementedException();
					}
					break;
				}
				case MathDomain::Int:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							throw NotImplementedException();
						case MemorySpace::Test:
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

		extern void DestroyCsrHandle(SparseMemoryTile& A)
		{
			switch (A.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::DestroyCsrHandle<MathDomain::Float>(A);
							break;
						case MemorySpace::Test:
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
							mkr::DestroyCsrHandle<MathDomain::Double>(A);
							break;
						case MemorySpace::Test:
							break;
						default:
							throw NotImplementedException();
					}
					break;
				}
				case MathDomain::Int:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							throw NotImplementedException();
						case MemorySpace::Test:
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
		 * zDense = alpha * xSparse + yDense
		 */
		void SparseAdd(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y, const double alpha)
		{
			assert(z.memorySpace == x.memorySpace);
			assert(z.memorySpace == y.memorySpace);
			assert(z.mathDomain == x.mathDomain);
			assert(z.mathDomain == y.mathDomain);

			switch (z.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (z.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::SparseAdd<MathDomain::Float>(z, x, y, alpha);
							break;

						case MemorySpace::Test:
						{
							const auto* iPtr = reinterpret_cast<const int*>(x.indices);	   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
							auto* zPtr = GetPointer<MathDomain::Float>(z);
							auto* xPtr = GetPointer<MathDomain::Float>(x);
							auto* yPtr = GetPointer<MathDomain::Float>(y);

							for (size_t i = 0; i < x.size; ++i)
								zPtr[iPtr[i]] = static_cast<float>(alpha) * xPtr[i] + yPtr[iPtr[i]];
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
							mkr::SparseAdd<MathDomain::Double>(z, x, y, alpha);
							break;

						case MemorySpace::Test:
						{
							const auto* iPtr = reinterpret_cast<const int*>(x.indices);	   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
							auto* zPtr = GetPointer<MathDomain::Double>(z);
							auto* xPtr = GetPointer<MathDomain::Double>(x);
							auto* yPtr = GetPointer<MathDomain::Double>(y);

							for (size_t i = 0; i < x.size; ++i)
								zPtr[iPtr[i]] = alpha * xPtr[i] + yPtr[iPtr[i]];
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
						case MemorySpace::Mkl:	  // TODO
						{
							const auto* iPtr = reinterpret_cast<const int*>(x.indices);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
							auto* zPtr = GetPointer<MathDomain::Int>(z);
							auto* xPtr = GetPointer<MathDomain::Int>(x);
							auto* yPtr = GetPointer<MathDomain::Int>(y);

							for (size_t i = 0; i < x.size; ++i)
								zPtr[iPtr[i]] = static_cast<int>(alpha) * xPtr[i] + yPtr[iPtr[i]];
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
		 * zDense = yDense - xSparse
		 */
		void SparseSubtract(MemoryBuffer& z, const SparseMemoryBuffer& x, const MemoryBuffer& y) { SparseAdd(z, x, y, -1.0); }

		/**
		 *	yDense = ASparse * xDense
		 */
		void SparseDot(MemoryBuffer& y, SparseMemoryTile& A, const MemoryBuffer& x, const MatrixOperation aOperation, const double alpha, const double beta)
		{
			switch (A.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::SparseDot<MathDomain::Float>(y, A, x, aOperation, alpha, beta);
							break;

						case MemorySpace::Test:
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
							mkr::SparseDot<MathDomain::Double>(y, A, x, aOperation, alpha, beta);
							break;

						case MemorySpace::Test:
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
						case MemorySpace::Mkl:
						default:
							throw NotImplementedException();
					}
				}
				default:
					throw NotImplementedException();
			}
		}

		/**
		 *	ADense = BSparse * CDense
		 */
		void SparseMultiply(MemoryTile& A, SparseMemoryTile& B, const MemoryTile& C, const MatrixOperation bOperation, const double alpha)
		{
			switch (A.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::SparseMultiply<MathDomain::Float>(A, B, C, bOperation, alpha);
							break;

						case MemorySpace::Test:
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
							mkr::SparseMultiply<MathDomain::Double>(A, B, C, bOperation, alpha);
							break;

						case MemorySpace::Test:
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
						case MemorySpace::Mkl:
						default:
							throw NotImplementedException();
					}
				}
				default:
					throw NotImplementedException();
			}
		}

		void SparseSolve(SparseMemoryTile& A, MemoryTile& B, LinearSystemSolverType solver)
		{
			switch (A.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (A.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::SparseSolve<MathDomain::Float>(A, B, solver);
							break;

						case MemorySpace::Test:
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
							mkr::SparseSolve<MathDomain::Double>(A, B, solver);
							break;

						case MemorySpace::Test:
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
						case MemorySpace::Mkl:
						default:
							throw NotImplementedException();
					}
				}
				default:
					throw NotImplementedException();
			}
		}
	}	 // namespace routines
}	 // namespace cl
