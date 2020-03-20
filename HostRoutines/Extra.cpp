
#include <Extra.h>
#include <Common.h>
#include <Exceptions.h>

#include <BlasWrappers.h>

#include <numeric>
#include <cmath>


namespace cl { namespace routines {

	void Sum(double& sum, const MemoryBuffer& v)
	{
		switch (v.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *vPtr = GetPointer<MathDomain::Float>(v);

				switch (v.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *vPtr = GetPointer<MathDomain::Double>(v);

				switch (v.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *vPtr = GetPointer<MathDomain::Int>(v);

				switch (v.memorySpace)
				{
					case MemorySpace::Test:
					{
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
					auto *xPtr = GetPointer<MathDomain::Float>(x);

					switch (x.memorySpace)
					{
						case MemorySpace::Test:
						{
							auto _min = xPtr[0];
							for (size_t i = 1; i < x.size; ++i)
							{
								if (xPtr[i] < _min)
									_min = xPtr[i];
							}
							min = static_cast<double>(_min);
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
							min = xPtr[0];
							for (size_t i = 1; i < x.size; ++i)
							{
								if (xPtr[i] < min)
									min = xPtr[i];
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
							auto _min = xPtr[0];
							for (size_t i = 1; i < x.size; ++i)
							{
								if (xPtr[i] < _min)
									_min = xPtr[i];
							}
							min = static_cast<double>(_min);
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
		switch (x.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
						auto _max = xPtr[0];
						for (size_t i = 1; i < x.size; ++i)
						{
							if (xPtr[i] > _max)
								_max = xPtr[i];
						}
						max = static_cast<double>(_max);
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
						max = xPtr[0];
						for (size_t i = 1; i < x.size; ++i)
						{
							if (xPtr[i] > max)
								max = xPtr[i];
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
						auto _max = xPtr[0];
						for (size_t i = 1; i < x.size; ++i)
						{
							if (xPtr[i] > _max)
								_max = xPtr[i];
						}
						max = static_cast<double>(_max);
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

	void AbsMin(double& min, const MemoryBuffer& x)
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
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *xPtr = GetPointer<MathDomain::Float>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *xPtr = GetPointer<MathDomain::Double>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
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
				auto *xPtr = GetPointer<MathDomain::Int>(x);

				switch (x.memorySpace)
				{
					case MemorySpace::Test:
					{
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
			default:
				throw NotImplementedException();
		}
	}

}}