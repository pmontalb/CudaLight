#include <Common.h>
#include <Exceptions.h>
#include <ForgeHelpers.h>

#include <algorithm>

namespace cl
{
	namespace routines
	{
		void MakePair(MemoryBuffer& z, const MemoryBuffer& x, const MemoryBuffer& y)
		{
			assert(z.mathDomain == MathDomain::Float);
			assert(x.size == y.size);
			assert(z.size == 2 * x.size);

			const auto makePair = [](auto* z_, const auto i, const auto* x_, const auto* y_) {
				const auto idx = i << 1;
				z_[idx] = static_cast<float>(x_[i]);
				z_[idx + 1] = static_cast<float>(y_[i]);
			};

			auto* zPtr = GetPointer<MathDomain::Float>(z);

			switch (x.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Float>(x);
							auto* yPtr = GetPointer<MathDomain::Float>(y);

							for (size_t i = 0; i < z.size; ++i)
								makePair(zPtr, i, xPtr, yPtr);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Double>(x);
							auto* yPtr = GetPointer<MathDomain::Double>(y);

							for (size_t i = 0; i < z.size; ++i)
								makePair(zPtr, i, xPtr, yPtr);
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
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Int>(x);
							auto* yPtr = GetPointer<MathDomain::Int>(y);

							for (size_t i = 0; i < z.size; ++i)
								makePair(zPtr, i, xPtr, yPtr);
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

		void MakeTriple(MemoryBuffer& v, const MemoryBuffer& x, const MemoryBuffer& y, const MemoryBuffer& z)
		{
			assert(v.mathDomain == MathDomain::Float);
			assert(x.size * y.size == z.size);
			assert(v.size == 3 * z.size);

			const auto makeTriple = [&](auto* v_, const auto* x_, const auto* y_, const auto* z_) {
				for (size_t i = 0; i < x.size; ++i)
				{
					for (size_t j = 0; j < y.size; ++j)
					{
						const size_t offset = j + i * y.size;
						v_[3 * offset] = static_cast<float>(x_[i]);
						v_[3 * offset + 1] = static_cast<float>(y_[j]);
						v_[3 * offset + 2] = static_cast<float>(z_[i + j * x.size]);
					}
				}
			};

			auto* vPtr = GetPointer<MathDomain::Float>(v);

			switch (x.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Float>(x);
							auto* yPtr = GetPointer<MathDomain::Float>(y);
							auto* zPtr = GetPointer<MathDomain::Float>(y);

							makeTriple(vPtr, xPtr, yPtr, zPtr);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Double>(x);
							auto* yPtr = GetPointer<MathDomain::Double>(y);
							auto* zPtr = GetPointer<MathDomain::Double>(y);

							makeTriple(vPtr, xPtr, yPtr, zPtr);
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
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Int>(x);
							auto* yPtr = GetPointer<MathDomain::Int>(y);
							auto* zPtr = GetPointer<MathDomain::Int>(y);

							makeTriple(vPtr, xPtr, yPtr, zPtr);
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

	}	 // namespace routines
}	 // namespace cl
