#include "stdafx.h"
#include "CppUnitTest.h"

#include <Tensor.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(TensorTests)
	{
	public:

		TEST_METHOD(Allocation)
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

		TEST_METHOD(Copy)
		{
			cl::GpuSingleTensor t1(10, 5, 5, 1.2345f);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuSingleTensor t2(t1);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(t1 == t2);

			cl::GpuDoubleTensor t3(10, 5, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleTensor t4(t3);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(t3 == t4);

			cl::GpuIntegerTensor v5(10, 5, 5, 10);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuIntegerTensor v6(v5);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(v5 == v6);
		}

		TEST_METHOD(Linspace)
		{
			cl::ten v = cl::LinSpace(0.0f, 1.0f, 10, 10, 10);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			Assert::IsTrue(fabs(_v[0] - 0.0) <= 1e-7);
			Assert::IsTrue(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
		}

		TEST_METHOD(RandomUniform)
		{
			cl::ten v = cl::RandomUniform(10, 10, 10, 1234);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			for (const auto& iter : _v)
				Assert::IsTrue(iter >= 0.0 && iter <= 1.0);
		}

		TEST_METHOD(RandomGaussian)
		{
			cl::ten v = cl::RandomGaussian(10, 10, 10, 1234);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			for (size_t i = 0; i < _v.size() / 2; ++i)
				Assert::IsTrue(fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7);
		}
	};
}