
#include <BufferInitializer.h>
#include <Common.h>
#include <Exceptions.h>

#include "MklAllWrappers.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <random>

namespace cl
{
	namespace routines
	{
		void Zero(MemoryBuffer& buf)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::memset(ptr, 0, buf.TotalSize());
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::memset(ptr, 0, buf.TotalSize());
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							std::memset(ptr, 0, buf.TotalSize());
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

		void Initialize(MemoryBuffer& buf, const double value)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::fill(ptr, ptr + buf.size, value);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::fill(ptr, ptr + buf.size, value);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:
						case MemorySpace::OpenBlas:
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							std::fill(ptr, ptr + buf.size, value);
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

		void Reciprocal(MemoryBuffer& buf)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<float>(), 1.0f));
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<double>(), 1.0));
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<int>(), 1));
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

		void LinSpace(MemoryBuffer& buf, const double x0, const double x1)
		{
			const auto linspaceWorker = [&](auto* begin, auto* end, auto val) {
				auto dx = (x1 - x0) / static_cast<double>(buf.size - 1);
				size_t i = 0;
				for (auto* iter = begin; iter != end; ++iter, ++i)
					*iter = static_cast<decltype(val)>(x0 + static_cast<double>(i) * dx);
			};
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							linspaceWorker(ptr, ptr + buf.size, 0.0f);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							linspaceWorker(ptr, ptr + buf.size, 0.0);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							linspaceWorker(ptr, ptr + buf.size, 0);
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

		void RandUniform(MemoryBuffer& buf, const unsigned seed)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::RandUniform<MathDomain::Float>(buf, seed);
							break;

						case MemorySpace::Test:
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);
							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::uniform_real_distribution<float> uniformDistribution { 0.0f, 1.0f };
							auto generator = [&uniformDistribution, &mersenneEngine]() { return uniformDistribution(mersenneEngine); };

							std::generate(ptr, ptr + buf.size, generator);
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
							mkr::RandUniform<MathDomain::Double>(buf, seed);
							break;

						case MemorySpace::Test:
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::uniform_real_distribution<double> uniformDistribution { 0.0, 1.0 };
							auto generator = [&uniformDistribution, &mersenneEngine]() { return uniformDistribution(mersenneEngine); };

							std::generate(ptr, ptr + buf.size, generator);
							break;
						}
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

		void RandNormal(MemoryBuffer& buf, const unsigned seed)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Mkl:
							mkr::RandNormal<MathDomain::Float>(buf, seed);
							break;

						case MemorySpace::Test:
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::normal_distribution<float> normalDistribution { 0.0f, 1.0f };
							auto generator = [&normalDistribution, &mersenneEngine]() { return normalDistribution(mersenneEngine); };

							std::generate(ptr, ptr + buf.size, generator);
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
							mkr::RandNormal<MathDomain::Double>(buf, seed);
							break;

						case MemorySpace::Test:
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::normal_distribution<double> normalDistribution { 0.0, 1.0 };
							auto generator = [&normalDistribution, &mersenneEngine]() { return normalDistribution(mersenneEngine); };

							std::generate(ptr, ptr + buf.size, generator);
							break;
						}
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

		void Eye(MemoryTile& buf)
		{
			assert(buf.nRows == buf.nCols);

			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
								ptr[i + i * buf.nRows] = 1.0f;
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
								ptr[i + i * buf.nRows] = 1.0;
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
								ptr[i + i * buf.nRows] = 1;
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

		void OnesUpperTriangular(MemoryTile& buf)
		{
			assert(buf.nRows == buf.nCols);

			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
							{
								for (size_t j = i; j < buf.nCols; ++j)
									ptr[i + j * buf.nRows] = 1.0f;
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
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
							{
								for (size_t j = i; j < buf.nCols; ++j)
									ptr[j + i * buf.nRows] = 1.0;
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
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							Zero(buf);
							for (size_t i = 0; i < buf.nRows; ++i)
							{
								for (size_t j = i; j < buf.nCols; ++j)
									ptr[j + i * buf.nRows] = 1;
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

		void RandShuffle(MemoryBuffer& buf, const unsigned seed)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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

		void RandShufflePair(MemoryBuffer& buf1, MemoryBuffer& buf2, const unsigned seed)
		{
			assert(buf1.memorySpace == buf2.memorySpace);

			switch (buf1.memorySpace)
			{
				case MemorySpace::Test:
				case MemorySpace::Mkl:		   // TODO
				case MemorySpace::OpenBlas:	   // TODO
				case MemorySpace::GenericBlas:
					RandShuffle(buf1, seed);
					RandShuffle(buf2, seed);
					break;
				default:
					throw NotImplementedException();
			}
		}

		void RandShuffleColumns(MemoryTile& buf, const unsigned seed)
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Float>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							// Fisher-Yates
							for (int j = static_cast<int>(buf.nCols) - 1; j > 0; j--)
							{
								// Pick a random index from 0 to i
								std::uniform_int_distribution<int> uniformDistribution { 0, j + 1 };

								const auto k = uniformDistribution(mersenneEngine);

								// Swap j-th column with k-th one
								for (size_t i = 0; i < buf.nRows; ++i)
									std::swap(ptr[i + static_cast<size_t>(j) * buf.nRows], ptr[i + static_cast<size_t>(k) * buf.nRows]);
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
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Double>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							// Fisher-Yates
							for (int j = static_cast<int>(buf.nCols) - 1; j > 0; j--)
							{
								// Pick a random index from 0 to i
								std::uniform_int_distribution<int> uniformDistribution { 0, j + 1 };

								const auto k = uniformDistribution(mersenneEngine);

								// Swap j-th column with k-th one
								for (size_t i = 0; i < buf.nRows; ++i)
									std::swap(ptr[i + static_cast<size_t>(j) * buf.nRows], ptr[i + static_cast<size_t>(k) * buf.nRows]);
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
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
						case MemorySpace::Mkl:		   // TODO
						case MemorySpace::OpenBlas:	   // TODO
						case MemorySpace::GenericBlas:
						{
							auto* ptr = GetPointer<MathDomain::Int>(buf);

							std::random_device randomDevice;
							std::mt19937 mersenneEngine { randomDevice() };
							mersenneEngine.seed(seed);

							// Fisher-Yates
							for (int j = static_cast<int>(buf.nCols) - 1; j > 0; j--)
							{
								// Pick a random index from 0 to i
								std::uniform_int_distribution<int> uniformDistribution { 0, j + 1 };

								const auto k = uniformDistribution(mersenneEngine);

								// Swap j-th column with k-th one
								for (size_t i = 0; i < buf.nRows; ++i)
									std::swap(ptr[i + static_cast<size_t>(j) * buf.nRows], ptr[i + static_cast<size_t>(k) * buf.nRows]);
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

		void RandShuffleColumnsPair(MemoryTile& buf1, MemoryTile& buf2, const unsigned seed)
		{
			assert(buf1.memorySpace == buf2.memorySpace);

			switch (buf1.memorySpace)
			{
				case MemorySpace::Test:
				case MemorySpace::Mkl:		   // TODO
				case MemorySpace::OpenBlas:	   // TODO
				case MemorySpace::GenericBlas:
					RandShuffleColumns(buf1, seed);
					RandShuffleColumns(buf2, seed);
					break;
				default:
					throw NotImplementedException();
			}
		}
	}	 // namespace routines
}	 // namespace cl
