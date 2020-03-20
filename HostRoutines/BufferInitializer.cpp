
#include <BufferInitializer.h>
#include <Exceptions.h>
#include <Common.h>

#include <cstring>
#include <algorithm>
#include <functional>
#include <random>

namespace cl { namespace routines {

	void Zero(MemoryBuffer &buf)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						std::memset(ptr, 0, buf.TotalSize());
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
						std::memset(ptr, 0, buf.TotalSize());
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
						std::memset(ptr, 0, buf.TotalSize());
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

	void Initialize(MemoryBuffer &buf, const double value)
	{
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						std::fill(ptr, ptr + buf.size, value);
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
						std::fill(ptr, ptr + buf.size, value);
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
						std::fill(ptr, ptr + buf.size, value);
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

	void Reciprocal(MemoryBuffer &buf)
	{
		{
			switch (buf.mathDomain)
			{
				case MathDomain::Float:
				{
					auto *ptr = GetPointer<MathDomain::Float>(buf);
					switch (buf.memorySpace)
					{
						case MemorySpace::Test:
							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<float>(), 1.0f));
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
							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<double>(), 1.0));
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
							std::transform(ptr, ptr + buf.size, ptr, std::bind1st(std::divides<int>(), 1));
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
	}

	void LinSpace(MemoryBuffer &buf, const double x0, const double x1)
	{
		const auto linspaceWorker = [&](auto* begin, auto* end, auto val)
		{
			auto dx = (x1 - x0) / static_cast<double>(buf.size - 1);
			size_t i = 0;
			for (auto* iter = begin; iter != end; ++iter, ++i)
				*iter = static_cast<decltype(val)>(x0 + static_cast<double>(i) * dx);
		};
		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						linspaceWorker(ptr, ptr + buf.size, 0.0f);
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
						linspaceWorker(ptr, ptr + buf.size, 0.0);
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
						linspaceWorker(ptr, ptr + buf.size, 0);
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

	void RandUniform(MemoryBuffer &buf, const unsigned seed)
	{
		std::random_device randomDevice;
		std::mt19937 mersenneEngine { randomDevice() };
		mersenneEngine.seed(seed);

		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				std::uniform_real_distribution<float> uniformDistribution {0.0f, 1.0f};
				auto generator = [&uniformDistribution, &mersenneEngine]() { return uniformDistribution(mersenneEngine); };

				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						std::generate(ptr, ptr + buf.size, generator);
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				std::uniform_real_distribution<double> uniformDistribution {0.0, 1.0};
				auto generator = [&uniformDistribution, &mersenneEngine]() { return uniformDistribution(mersenneEngine); };

				auto *ptr = GetPointer<MathDomain::Double>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						std::generate(ptr, ptr + buf.size, generator);
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

	void RandNormal(MemoryBuffer &buf, const unsigned seed)
	{
		std::random_device randomDevice;
		std::mt19937 mersenneEngine { randomDevice() };
		mersenneEngine.seed(seed);

		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				std::normal_distribution<float> normalDistribution {0.0f, 1.0f};

				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < (buf.size + 1) / 2; ++i)
						{
							auto rnd = normalDistribution(mersenneEngine);
							ptr[2 * i] = rnd;
							if (2 * i + 1 < buf.size)
								ptr[2 * i + 1] = -rnd;
						}
						break;
					default:
						throw NotImplementedException();
				}
				break;
			}
			case MathDomain::Double:
			{
				std::normal_distribution<double> normalDistribution {0.0, 1.0};

				auto *ptr = GetPointer<MathDomain::Double>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						for (size_t i = 0; i < (buf.size + 1) / 2; ++i)
						{
							auto rnd = normalDistribution(mersenneEngine);
							ptr[2 * i] = rnd;
							if (2 * i + 1 < buf.size)
								ptr[2 * i + 1] = -rnd;
						}
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

	void Eye(MemoryTile &buf)
	{
		assert(buf.nRows == buf.nCols);

		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
							ptr[i + i * buf.nRows] = 1.0f;
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
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
							ptr[i + i * buf.nRows] = 1.0;
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
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
							ptr[i + i * buf.nRows] = 1;
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

	void OnesUpperTriangular(MemoryTile &buf)
	{
		assert(buf.nRows == buf.nCols);

		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
						{
							for (size_t j = i; j < buf.nCols; ++j)
								ptr[i + j * buf.nRows] = 1.0f;
						}
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
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
						{
							for (size_t j = i; j < buf.nCols; ++j)
								ptr[j + i * buf.nRows] = 1.0;
						}
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
						Zero(buf);
						for (size_t i = 0; i < buf.nRows; ++i)
						{
							for (size_t j = i; j < buf.nCols; ++j)
								ptr[j + i * buf.nRows] = 1;
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

	void RandShuffle(MemoryBuffer &buf, const unsigned seed)
	{
		std::random_device randomDevice;
		std::mt19937 mersenneEngine { randomDevice() };
		mersenneEngine.seed(seed);

		switch (buf.mathDomain)
		{
			case MathDomain::Float:
			{
				auto *ptr = GetPointer<MathDomain::Float>(buf);
				switch (buf.memorySpace)
				{
					case MemorySpace::Test:
						std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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
						std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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
						std::shuffle(ptr, ptr + buf.size, mersenneEngine);
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

	void RandShufflePair(MemoryBuffer &buf1, MemoryBuffer &buf2, const unsigned seed)
	{
		assert(buf1.memorySpace == buf2.memorySpace);

		switch (buf1.memorySpace)
		{
			case MemorySpace::Test:
				RandShuffle(buf1, seed);
				RandShuffle(buf2, seed);
				break;
			default:
				throw NotImplementedException();
		}
	}

	void RandShuffleColumns(MemoryTile &, const unsigned)
	{
		throw NotImplementedException();
	}

	void RandShuffleColumnsPair(MemoryTile &, MemoryTile &, const unsigned )
	{
		throw NotImplementedException();
	}
}}