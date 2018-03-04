#include "stdafx.h"
#include "CppUnitTest.h"
#include <Vector.h>
#include <ColumnWiseMatrix.h>

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

		TEST_METHOD(AddMatrix)
		{
			cl::mat m1 = cl::LinSpace(-1.0, 1.0, 100, 100);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m1 = m1.Get();

			cl::mat m2 = cl::RandomUniform(m1.nRows(), m1.nCols(), 1234);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m2 = m2.Get();

			auto m3 = m1 + m2;
			dm::DeviceManager::CheckDeviceSanity();
			auto _m3 = m3.Get();

			for (size_t i = 0; i < m1.size(); ++i)
				Assert::IsTrue(fabs(_m3[i] - _m1[i] - _m2[i]) <= 1e-7);

			auto m4 = m1.Add(m2, 2.0);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m4 = m4.Get();

			for (size_t i = 0; i < m1.size(); ++i)
				Assert::IsTrue(fabs(_m4[i] - _m1[i] - 2.0 * _m2[i]) <= 1.2e-7);
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

		TEST_METHOD(Multiply)
		{
			cl::mat m1(10, 10, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m1 = m1.Get();

			cl::mat m2(10, 10, 9.8765);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m2 = m2.Get();

			auto m3 = m1 * m2;
			dm::DeviceManager::CheckDeviceSanity();
			auto _m3 = m3.Get();

			for (size_t i = 0; i < m1.nRows(); ++i)
			{
				for (size_t j = 0; j < m1.nCols(); ++j)
				{
					double m1m2 = 0.0;
					for (size_t k = 0; k < m1.nCols(); ++k)
						m1m2 += _m1[i + k * m1.nRows()] * _m2[k + j * m2.nRows()];
					Assert::IsTrue(fabs(m1m2 - _m3[i + j * m1.nRows()]) <= 5e-5);
				}
			}
		}

		TEST_METHOD(Dot)
		{
			cl::mat m1(10, 10, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();
			auto _m1 = m1.Get();

			cl::vec v1(10, 9.8765);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();

			auto v2 = m1 * v1;
			dm::DeviceManager::CheckDeviceSanity();
			auto _v2 = v2.Get();

			for (size_t i = 0; i < m1.nRows(); ++i)
			{
				double m1v1 = 0.0;
				for (size_t j = 0; j < m1.nCols(); ++j)
					m1v1 += _m1[i + j * m1.nRows()] * _v1[j];
				Assert::IsTrue(fabs(m1v1 - _v2[i]) <= 5e-5);
			}
		}
	};
}