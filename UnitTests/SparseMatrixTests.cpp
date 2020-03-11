#include <gtest/gtest.h>

#include <CompressedSparseRowMatrix.h>

namespace clt
{
	class SparseMatrixTests : public ::testing::Test
	{
	};

	TEST_F(SparseMatrixTests, Allocation)
	{
		std::vector<int> _NonZeroCols = { 0, 1, 1, 3, 2, 3, 4, 5 };
		cl::ivec gpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		gpuNonZeroCols.ReadFrom(_NonZeroCols);

		std::vector<int> _NonZeroRows = { 0, 2, 4, 7, 8 };
		cl::ivec gpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		gpuNonZeroRows.ReadFrom(_NonZeroRows);

		cl::GpuSingleSparseMatrix m1(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleSparseMatrix m2(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuIntegerVector cpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		cpuNonZeroCols.ReadFrom(_NonZeroCols);

		cl::CpuIntegerVector cpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		cpuNonZeroRows.ReadFrom(_NonZeroRows);
		cl::CpuSingleSparseMatrix m3(4, 6, cpuNonZeroCols, cpuNonZeroRows, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuDoubleSparseMatrix m4(4, 6, cpuNonZeroCols, cpuNonZeroRows, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(SparseMatrixTests, Copy)
	{
		std::vector<int> _NonZeroCols = { 0, 1, 1, 3, 2, 3, 4, 5 };
		cl::ivec gpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		gpuNonZeroCols.ReadFrom(_NonZeroCols);

		std::vector<int> _NonZeroRows = { 0, 2, 4, 7, 8 };
		cl::ivec gpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		gpuNonZeroRows.ReadFrom(_NonZeroRows);

		cl::GpuSingleSparseMatrix m1(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuSingleSparseMatrix m2(m1);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m1 == m2);

		cl::GpuDoubleSparseMatrix m3(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleSparseMatrix m4(m3);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m3 == m4);

		cl::GpuIntegerSparseMatrix m5(4, 6, gpuNonZeroCols, gpuNonZeroRows, 10);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuIntegerSparseMatrix m6(m5);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m5 == m6);
	}

	TEST_F(SparseMatrixTests, ReadFromDense)
	{
		std::vector<float> denseMatrix(24);
		denseMatrix[10] = 2.7182f;
		denseMatrix[20] = 3.1415f;
		denseMatrix[22] = 1.6180f;

		cl::mat dv(denseMatrix, 4, 6);
		cl::smat sv(dv);

		auto _dv = dv.Get();
		auto _sv = dv.Get();
		ASSERT_EQ(_dv.size(), _sv.size());

		for (size_t i = 0; i < _dv.size(); ++i)
		{
			ASSERT_TRUE(std::fabs(_dv[i] - _sv[i]) <= 1e-7f);
		}
	}
}
