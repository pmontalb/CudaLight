#include <gtest/gtest.h>

#include <Vector.h>

namespace clt
{
	class VectorTests : public ::testing::Test
	{
	};

	TEST_F(VectorTests,Allocation)
	{
		cl::GpuSingleVector v1(10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleVector v2(10, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuSingleVector v3(10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuDoubleVector v4(10, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(VectorTests,Copy)
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

	TEST_F(VectorTests,Linspace)
	{
		cl::vec v = cl::LinSpace(0.0, 1.0, 10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		ASSERT_TRUE(fabs(_v[0] - 0.0) <= 1e-7);
		ASSERT_TRUE(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
	}

	TEST_F(VectorTests,RandomUniform)
	{
		cl::vec v = cl::RandomUniform(10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (const auto& iter: _v)
			ASSERT_TRUE(iter >= 0.0 && iter <= 1.0);
	}

	TEST_F(VectorTests,RandomGaussian)
	{
		cl::vec v = cl::RandomGaussian(10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < _v.size() / 2; ++i)
			ASSERT_TRUE(fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7);
	}
}