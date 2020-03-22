
#include <Common.h>
#include <Flags.h>
#include <Types.h>
#include <Exceptions.h>
#include <MemoryManager.h>
#include "MklWrappers.h"

namespace cl { namespace routines {

	void Copy(MemoryBuffer& dest, const MemoryBuffer& source)
	{
		assert(dest.memorySpace == source.memorySpace);
		assert(dest.mathDomain == source.mathDomain);
		assert(dest.size == source.size);

		switch (dest.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (dest.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Copy<MathDomain::Float>(dest, source);
						break;

					case MemorySpace::Test:
					{
						auto *destPtr = GetPointer<MathDomain::Float>(dest);
						const auto *sourcePtr = GetPointer<MathDomain::Float>(source);

						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (dest.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Copy<MathDomain::Double>(dest, source);
						break;

					case MemorySpace::Test:
					{
						auto *destPtr = GetPointer<MathDomain::Double>(dest);
						const auto *sourcePtr = GetPointer<MathDomain::Double>(source);

						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (dest.memorySpace)
				{
					case MemorySpace::Test:
					case MemorySpace::Mkl:  // TODO
					{
						auto *destPtr = GetPointer<MathDomain::Int>(dest);
						const auto *sourcePtr = GetPointer<MathDomain::Int>(source);

						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
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

	void Alloc(MemoryBuffer& buf)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Alloc(buf);
						break;

					case MemorySpace::Test:
					{
						auto** ptr = GetRefPointer<MathDomain::Float>(buf);
						*ptr = new float[buf.size];
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Alloc(buf);
						break;

					case MemorySpace::Test:
					{
						auto* ptr = GetRefPointer<MathDomain::Double>(buf);
						*ptr = new double[buf.size];
						break;
					}

					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Alloc(buf);
						break;

					case MemorySpace::Test:
					{
						auto* ptr = GetRefPointer<MathDomain::Int>(buf);
						*ptr = new int[buf.size];
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

	void Free(MemoryBuffer& buf)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Free(buf);
						break;

					case MemorySpace::Test:
					{
						auto **ptr = GetRefPointer<MathDomain::Float>(buf);
						delete[] *ptr;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Free(buf);
						break;

					case MemorySpace::Test:
					{
						auto **ptr = GetRefPointer<MathDomain::Double>(buf);
						delete[] *ptr;
						break;
					}
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				switch (buf.memorySpace)
				{
					case MemorySpace::Mkl:
						mkr::Free(buf);
						break;

					case MemorySpace::Test:
					{
						auto *ptr = GetRefPointer<MathDomain::Int>(buf);
						delete[] *ptr;
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

		buf.pointer = 0;
	}
}}
