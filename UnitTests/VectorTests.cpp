#include <gtest/gtest.h>

#include <Vector.h>

#include <algorithm>

namespace clt
{
	class VectorTests : public ::testing::Test
	{
	};
	
	TEST_F(VectorTests, Allocation)
	{
		cl::GpuSingleVector v1(10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleVector v2(10, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CudaCpuSingleVector v3(10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CudaCpuDoubleVector v4(10, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(VectorTests, Copy)
	{
		cl::GpuSingleVector v1(10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuSingleVector v2(v1);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v1 == v2);

		cl::GpuDoubleVector v3(10, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleVector v4(v3);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v3 == v4);

		cl::GpuIntegerVector v5(10, 10);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuIntegerVector v6(v5);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(VectorTests, Linspace)
	{
		cl::vec v = cl::vec::LinSpace(0.0, 1.0, 10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(VectorTests, RandomUniform)
	{
		cl::vec v = cl::vec::RandomUniform(10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (const auto& iter: _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(VectorTests, RandomGaussian)
	{
		cl::vec v = cl::vec::RandomGaussian(10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < _v.size() / 2; ++i)
			ASSERT_TRUE(std::fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7f);
	}

	TEST_F(VectorTests, RandomShuffle)
	{
		cl::vec v = cl::vec::RandomGaussian(10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v.Get();

		cl::vec::RandomShuffle(v, 2345);
		auto _v2 = v.Get();
		auto _v3 = v.Get();

		std::sort(_v1.begin(), _v1.end());
		std::sort(_v2.begin(), _v2.end());
		for (size_t i = 0; i < _v2.size(); ++i)
		{
			ASSERT_DOUBLE_EQ(_v1[i], _v2[i]);
			ASSERT_NE(_v1[i], _v3[i]);
		}
	}

	TEST_F(VectorTests, RandomShufflePair)
	{
		cl::vec u = cl::vec::RandomGaussian(10, 1234);
		cl::vec v = cl::vec::RandomGaussian(10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _u1 = v.Get();
		auto _v1 = v.Get();

		cl::vec::RandomShufflePair(u, v, 2345);
		auto _u2 = u.Get();
		auto _u3 = u.Get();
		auto _v2 = v.Get();
		auto _v3 = v.Get();

		std::sort(_v1.begin(), _v1.end());
		std::sort(_v2.begin(), _v2.end());
		std::sort(_u1.begin(), _u1.end());
		std::sort(_u2.begin(), _u2.end());
		for (size_t i = 0; i < _v2.size(); ++i)
		{
			ASSERT_DOUBLE_EQ(_u1[i], _u2[i]);
			ASSERT_DOUBLE_EQ(_v1[i], _v2[i]);
			ASSERT_NE(_u1[i], _u3[i]);
			ASSERT_NE(_v1[i], _v3[i]);

			// check permutation used the same indices
			unsigned j = 0;
			bool found = false;
			for (; j < _u2.size(); ++j)
			{
				if (std::fabs(_u3[j] - _u1[i]) < 1e-12f)
				{
					found = true;
					break;
				}
			}
			ASSERT_TRUE(found);

			unsigned k = 0;
			found = false;
			for (; k < _v2.size(); ++k)
			{
				if (std::fabs(_v3[k] - _v1[i]) < 1e-12f)
				{
					found = true;
					break;
				}
			}
			ASSERT_TRUE(found);
			ASSERT_EQ(j, k);
		}
	}

	TEST_F(VectorTests, EuclideanNorm)
	{
		cl::vec u = cl::vec::RandomGaussian(10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _u = u.Get();
		
		auto norm = u.EuclideanNorm();
		float normCpu = 0.0;
		for (auto& x: _u)
			normCpu += x * x;
		ASSERT_NEAR(normCpu, norm * norm, 1e-6);
	}
}
