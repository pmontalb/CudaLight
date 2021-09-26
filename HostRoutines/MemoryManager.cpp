
#include <MemoryManager.h>

#include <Common.h>
#include <Exceptions.h>
#include <Flags.h>
#include <GenericBlasAllWrappers.h>
#include <MklAllWrappers.h>
#include <OpenBlasAllWrappers.h>
#include <Types.h>

#include <cstdlib>

namespace cl
{
	namespace routines
	{
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
						case MemorySpace::OpenBlas:
							obr::Copy<MathDomain::Float>(dest, source);
							break;
						case MemorySpace::GenericBlas:
							gbr::Copy<MathDomain::Float>(dest, source);
							break;

						case MemorySpace::Test:
						{
							auto* destPtr = GetPointer<MathDomain::Float>(dest);
							const auto* sourcePtr = GetPointer<MathDomain::Float>(source);

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
						case MemorySpace::OpenBlas:
							obr::Copy<MathDomain::Double>(dest, source);
							break;
						case MemorySpace::GenericBlas:
							gbr::Copy<MathDomain::Double>(dest, source);
							break;

						case MemorySpace::Test:
						{
							auto* destPtr = GetPointer<MathDomain::Double>(dest);
							const auto* sourcePtr = GetPointer<MathDomain::Double>(source);

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
						case MemorySpace::Mkl:			  // TODO
						case MemorySpace::OpenBlas:		  // TODO
						case MemorySpace::GenericBlas:	  // TODO
						{
							auto* destPtr = GetPointer<MathDomain::Int>(dest);
							const auto* sourcePtr = GetPointer<MathDomain::Int>(source);

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
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto** ptr = GetRefPointer<MathDomain::Float>(buf);

							static constexpr size_t alignmentBits = { 64 };
							auto err = posix_memalign(reinterpret_cast<void**>(ptr), alignmentBits, buf.TotalSize()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
							assert(err == 0);
							assert(*ptr != nullptr);

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
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetRefPointer<MathDomain::Double>(buf);

							static constexpr size_t alignmentBits = { 64 };
							auto err = posix_memalign(reinterpret_cast<void**>(ptr), alignmentBits, buf.TotalSize()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
							assert(err == 0);
							assert(*ptr != nullptr);
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
						case MemorySpace::OpenBlas:		  // TODO
						case MemorySpace::GenericBlas:	  // TODO
						{
							auto* ptr = GetRefPointer<MathDomain::Int>(buf);

							static constexpr size_t alignmentBits = { 64 };
							auto err = posix_memalign(reinterpret_cast<void**>(ptr), alignmentBits, buf.TotalSize()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
							assert(err == 0);
							assert(*ptr != nullptr);
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
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto** ptr = GetRefPointer<MathDomain::Float>(buf);
							std::free(*ptr);  // NOLINT
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
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto** ptr = GetRefPointer<MathDomain::Double>(buf);
							std::free(*ptr);  // NOLINT
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
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetRefPointer<MathDomain::Int>(buf);
							std::free(*ptr);  // NOLINT
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
	}	 // namespace routines
}	 // namespace cl
