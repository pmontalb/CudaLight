
#include <Common.h>
#include <Flags.h>
#include <Types.h>
#include <Exceptions.h>
#include <MemoryManager.h>

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
				auto *destPtr = GetPointer<MathDomain::Float>(dest);
				const auto *sourcePtr = GetPointer<MathDomain::Float>(source);
				switch (dest.memorySpace)
				{
					case MemorySpace::Test:
						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *destPtr = GetPointer<MathDomain::Double>(dest);
				const auto *sourcePtr = GetPointer<MathDomain::Double>(source);
				switch (dest.memorySpace)
				{
					case MemorySpace::Test:
						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *destPtr = GetPointer<MathDomain::Int>(dest);
				const auto *sourcePtr = GetPointer<MathDomain::Int>(source);
				switch (dest.memorySpace)
				{
					case MemorySpace::Test:
						std::copy(sourcePtr, sourcePtr + source.size, destPtr);
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

	void Alloc(MemoryBuffer& buf)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto** ptr = GetRefPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						*ptr = new float[buf.size];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto* ptr = GetRefPointer<MathDomain::Double>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						*ptr = new double[buf.size];
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto* ptr = GetRefPointer<MathDomain::Int>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						*ptr = new int[buf.size];
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

	void Free(MemoryBuffer& buf)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						delete[] ptr;
						buf.pointer = 0;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				auto *ptr = GetPointer<MathDomain::Double>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						delete[] ptr;
						buf.pointer = 0;
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Int:
			{
				auto *ptr = GetPointer<MathDomain::Int>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						delete[] ptr;
						buf.pointer = 0;
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