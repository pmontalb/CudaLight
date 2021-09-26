
#include <Common.h>
#include <Exceptions.h>
#include <Extra.h>

#include <BlasWrappers.h>

#include <algorithm>
#include <cmath>
#include <numeric>


namespace cl
{
	namespace routines
	{
		void Sum(double& sum, const MemoryBuffer& v)
		{
			switch (v.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (v.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* vPtr = GetPointer<MathDomain::Float>(v);

							sum = std::accumulate(vPtr, vPtr + v.size, 0.0);
							break;
						}
						default:
							throw NotImplementedException();
					}
					break;
				}
				case MathDomain::Double:
				{
					switch (v.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* vPtr = GetPointer<MathDomain::Double>(v);

							sum = std::accumulate(vPtr, vPtr + v.size, 0.0);
							break;
						}
						default:
							throw NotImplementedException();
					}
					break;
				}
				case MathDomain::Int:
				{
					switch (v.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* vPtr = GetPointer<MathDomain::Int>(v);

							sum = std::accumulate(vPtr, vPtr + v.size, 0);
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

		void Min(double& min, const MemoryBuffer& x)
		{
			switch (x.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Float>(x);
							min = static_cast<double>(*std::min_element(xPtr, xPtr + x.size));
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
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Double>(x);
							min = *std::min_element(xPtr, xPtr + x.size);
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
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Int>(x);
							min = static_cast<double>(*std::min_element(xPtr, xPtr + x.size));
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

		void Max(double& max, const MemoryBuffer& x)
		{
			{
				switch (x.mathDomain)
				{
					case MathDomain::Float:
					{
						switch (x.memorySpace)
						{
							case MemorySpace::Test:
							case MemorySpace::Mkl:		   // TODO
							case MemorySpace::OpenBlas:	   // TODO
							case MemorySpace::GenericBlas:
							{
								auto* xPtr = GetPointer<MathDomain::Float>(x);
								max = static_cast<double>(*std::max_element(xPtr, xPtr + x.size));
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
							case MemorySpace::Mkl:		   // TODO
							case MemorySpace::OpenBlas:	   // TODO
							case MemorySpace::GenericBlas:
							{
								auto* xPtr = GetPointer<MathDomain::Double>(x);
								max = *std::max_element(xPtr, xPtr + x.size);
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
							case MemorySpace::Mkl:		   // TODO
							case MemorySpace::OpenBlas:	   // TODO
							case MemorySpace::GenericBlas:
							{
								auto* xPtr = GetPointer<MathDomain::Int>(x);
								max = static_cast<double>(*std::max_element(xPtr, xPtr + x.size));
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
		}

		void AbsMin(double& min, const MemoryBuffer& x)
		{
			switch (x.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Float>(x);

							int argMin = -1;
							ArgAbsMin(argMin, x);
							min = static_cast<double>(std::fabs(xPtr[static_cast<size_t>(argMin)]));
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
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Double>(x);

							int argMin = -1;
							ArgAbsMin(argMin, x);
							min = std::fabs(xPtr[static_cast<size_t>(argMin)]);
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
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Int>(x);

							int argMin = -1;
							ArgAbsMin(argMin, x);
							min = std::fabs(xPtr[static_cast<size_t>(argMin)]);
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

		void AbsMax(double& max, const MemoryBuffer& x)
		{
			switch (x.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* xPtr = GetPointer<MathDomain::Float>(x);

							int argMax = -1;
							ArgAbsMax(argMax, x);
							max = static_cast<double>(std::fabs(xPtr[static_cast<size_t>(argMax)]));
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
						case MemorySpace::Mkl:			  // TODO
						case MemorySpace::OpenBlas:		  // TODO
						case MemorySpace::GenericBlas:	  // TODO
						{
							auto* xPtr = GetPointer<MathDomain::Double>(x);

							int argMax = -1;
							ArgAbsMax(argMax, x);
							max = std::fabs(xPtr[static_cast<size_t>(argMax)]);
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
						case MemorySpace::Mkl:			  // TODO
						case MemorySpace::OpenBlas:		  // TODO
						case MemorySpace::GenericBlas:	  // TODO
						{
							auto* xPtr = GetPointer<MathDomain::Int>(x);

							int argMin = -1;
							ArgAbsMax(argMin, x);
							max = std::fabs(xPtr[static_cast<size_t>(argMin)]);
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
