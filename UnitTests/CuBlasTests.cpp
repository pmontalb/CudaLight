#include "stdafx.h"
#include "CppUnitTest.h"
#include <Vector.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(CuBlasTests)
	{
	public:
		TEST_METHOD(Add)
		{
			cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();

			cl::vec v2 = cl::RandomUniform(v1.size());
			dm::DeviceManager::CheckDeviceSanity();
			auto _v2 = v2.Get();

			auto v3 = v1 + v2;
			dm::DeviceManager::CheckDeviceSanity();
			auto _v3 = v3.Get();

			for (size_t i = 0; i < v1.size(); ++i)
				Assert::IsTrue(fabs(_v3[i] - _v1[i] - _v2[i]) <= 1e-7);

			auto v4 = v1.Add(v2, 2.0);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v4 = v4.Get();

			for (size_t i = 0; i < v1.size(); ++i)
				Assert::IsTrue(fabs(_v4[i] - _v1[i] - 2.0 * _v2[i]) <= 1.2e-7);
		}

		TEST_METHOD(Scale)
		{
			cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();

			v1.Scale(2.0);
			auto _v2 = v1.Get();

			for (size_t i = 0; i < v1.size(); ++i)
				Assert::IsTrue(fabs(2.0 * _v1[i] - _v2[i]) <= 1e-7);
		}

		TEST_METHOD(ElementWiseProduct)
		{
			cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();

			cl::vec v2 = cl::RandomUniform(v1.size());
			dm::DeviceManager::CheckDeviceSanity();
			auto _v2 = v2.Get();

			auto v3 = v1 % v2;
			auto _v3 = v3.Get();

			for (size_t i = 0; i < v1.size(); ++i)
				Assert::IsTrue(fabs(_v3[i] - _v1[i] * _v2[i]) <= 1e-7);
		}

	};
}