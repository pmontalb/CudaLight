#include <gtest/gtest.h>

#include <Tensor.h>

namespace clt
{
	class TensorTests : public ::testing::Test
	{
	};

	TEST_F(TensorTests,Allocation)
	{
		cl::GpuSingleTensor t1(10, 5, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleTensor t2(10, 5, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuSingleTensor t3(10, 5, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuDoubleTensor t4(10, 5, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(TensorTests,Copy)
	{
		cl::GpuSingleTensor t1(10, 5, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuSingleTensor t2(t1);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(t1 == t2);

		cl::GpuDoubleTensor t3(10, 5, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleTensor t4(t3);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(t3 == t4);

		cl::GpuIntegerTensor v5(10, 5, 5, 10);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuIntegerTensor v6(v5);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(TensorTests,Linspace)
	{
		cl::ten v = cl::LinSpace(0.0f, 1.0f, 10, 10, 10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		ASSERT_TRUE(fabs(_v[0] - 0.0) <= 1e-7);
		ASSERT_TRUE(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
	}

	TEST_F(TensorTests,RandomUniform)
	{
		cl::ten v = cl::RandomUniform(10, 10, 10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0 && iter <= 1.0);
	}

	TEST_F(TensorTests,RandomGaussian)
	{
		cl::ten v = cl::RandomGaussian(10, 10, 10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < _v.size() / 2; ++i)
			ASSERT_TRUE(fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7);
	}
}