#include "stdafx.h"
#include "CppUnitTest.h"
#include <ColumnWiseMatrix.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(MatrixTests)
	{
	public:

		TEST_METHOD(Allocation)
		{
			cl::GpuSingleMatrix m1(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleMatrix m2(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::CpuSingleMatrix m3(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::CpuDoubleMatrix m4(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();
		}

		TEST_METHOD(Copy)
		{
			cl::GpuSingleMatrix m1(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuSingleMatrix m2(m1);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(m1 == m2);

			cl::GpuDoubleMatrix m3(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleMatrix m4(m3);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(m3 == m4);

			cl::GpuIntegerMatrix m5(10, 5, 10);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuIntegerMatrix m6(m5);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(m5 == m6);
		}

		TEST_METHOD(Linspace)
		{
			cl::mat v = cl::LinSpace(0.0, 1.0, 10, 10);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			Assert::IsTrue(fabs(_v[0] - 0.0) <= 1e-7);
			Assert::IsTrue(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
		}

		TEST_METHOD(RandomUniform)
		{
			cl::mat v = cl::RandomUniform(10, 10, 1234);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			for (const auto& iter : _v)
				Assert::IsTrue(iter >= 0.0 && iter <= 1.0);
		}

		TEST_METHOD(RandomGaussian)
		{
			cl::mat v = cl::RandomGaussian(10, 10, 1234);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v = v.Get();
			for (size_t i = 0; i < _v.size() / 2; ++i)
				Assert::IsTrue(fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7);
		}

		TEST_METHOD(GetColumn)
		{
			cl::mat m1(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			for (unsigned j = 0; j < m1.nCols(); ++j)
			{
				auto col = m1.Get(j);
				dm::DeviceManager::CheckDeviceSanity();
				
				Assert::AreEqual((unsigned)(col.size()), m1.nRows());
				for (size_t i = 0; i < col.size(); ++i)
					Assert::IsTrue(fabs(col[i] - 1.2345) <= 1e-7);
			}

		}

		TEST_METHOD(SetColumn)
		{
			cl::mat m1(10, 5, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m1 = m1.Get();

			const cl::vec v1(10, 2.3456);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();
			m1.Set(v1, 3);

			for (unsigned j = 0; j < m1.nCols(); ++j)
			{
				auto col = m1.Get(j);
				dm::DeviceManager::CheckDeviceSanity();

				Assert::AreEqual((unsigned)(col.size()), m1.nRows());

				if (j != 3)
				{
					for (size_t i = 0; i < col.size(); ++i)
						Assert::IsTrue(fabs(col[i] - _m1[i + m1.nRows() * j]) <= 1e-7);
				}
				else
				{
					for (size_t i = 0; i < col.size(); ++i)
						Assert::IsTrue(fabs(col[i] - _v1[i]) <= 1e-7);
				}
			}

		}
	};
}