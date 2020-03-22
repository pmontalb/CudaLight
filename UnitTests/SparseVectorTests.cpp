#include <gtest/gtest.h>
#include <SparseVector.h>

namespace clt
{
	class SparseVectorTests : public ::testing::Test
	{
	};

	TEST_F(SparseVectorTests, Allocation)
	{
		std::vector<int> indices = { 0, 5 };
		cl::ivec gpuIndices(static_cast<unsigned>(indices.size()));
		gpuIndices.ReadFrom(indices);

		cl::GpuSingleSparseVector v1(10, gpuIndices, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleSparseVector v2(10, gpuIndices, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::cudaCpu::ivec cpuIndices(static_cast<unsigned>(indices.size()));
		cpuIndices.ReadFrom(indices);
		cl::CpuSingleSparseVector v3(10, cpuIndices, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuDoubleSparseVector v4(10, cpuIndices, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(SparseVectorTests,Copy)
	{
		std::vector<int> indices = { 0, 5 };
		cl::ivec gpuIndices(static_cast<unsigned>(indices.size()));
		gpuIndices.ReadFrom(indices);

		cl::GpuSingleSparseVector v1(10, gpuIndices, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuSingleSparseVector v2(v1);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v1 == v2);

		cl::GpuDoubleSparseVector v3(10, gpuIndices, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleSparseVector v4(v3);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v3 == v4);

		cl::GpuIntegerSparseVector v5(10, gpuIndices, 10);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuIntegerSparseVector v6(v5);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(SparseVectorTests,ReadFromDense)
	{
		std::vector<float> denseVector(50);
		denseVector[10] = 2.7182f;
		denseVector[20] = 3.1415f;
		denseVector[30] = 1.6180f;

		cl::vec dv(denseVector);
		cl::svec sv(dv);

		auto _dv = dv.Get();
		auto _sv = dv.Get();
		ASSERT_EQ(_dv.size(), _sv.size());

		for (size_t i = 0; i < _dv.size(); ++i)
		{
			ASSERT_TRUE(std::fabs(_dv[i] - _sv[i]) <= 1e-7f);
		}
	}
}
