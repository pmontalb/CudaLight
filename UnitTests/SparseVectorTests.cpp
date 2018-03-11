#include "stdafx.h"
#include "CppUnitTest.h"

#include <SparseVector.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(SparseVectorTests)
	{
	public:

		TEST_METHOD(Allocation)
		{
			std::vector<int> indices = { 0, 5 };
			cl::ivec gpuIndices((unsigned)indices.size());
			gpuIndices.ReadFrom(indices);

			cl::GpuSingleSparseVector v1(10, gpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleSparseVector v2(10, gpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::CpuIntegerVector cpuIndices((unsigned)indices.size());
			cpuIndices.ReadFrom(indices);
			cl::CpuSingleSparseVector v3(10, cpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::CpuDoubleSparseVector v4(10, cpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();
		}

		TEST_METHOD(Copy)
		{
			std::vector<int> indices = { 0, 5 };
			cl::ivec gpuIndices((unsigned)indices.size());
			gpuIndices.ReadFrom(indices);

			cl::GpuSingleSparseVector v1(10, gpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuSingleSparseVector v2(v1);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(v1 == v2);

			cl::GpuDoubleSparseVector v3(10, gpuIndices, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleSparseVector v4(v3);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(v3 == v4);

			cl::GpuIntegerSparseVector v5(10, gpuIndices, 10);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuIntegerSparseVector v6(v5);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(v5 == v6);
		}

		TEST_METHOD(ReadFromDense)
		{
			std::vector<float> denseVector(50);
			denseVector[10] = 2.7182;
			denseVector[20] = 3.1415;
			denseVector[30] = 1.6180;

			cl::vec dv(denseVector);
			cl::svec sv(dv);

			auto _dv = dv.Get();
			auto _sv = dv.Get();
			Assert::AreEqual(_dv.size(), _sv.size());

			for (size_t i = 0; i < _dv.size(); ++i)
			{
				Assert::IsTrue(fabs(_dv[i] - _sv[i]) <= 1e-7);
			}
		}
	};
}