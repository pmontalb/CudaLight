#include "stdafx.h"
#include "CppUnitTest.h"

#include <CompressedSparseRowMatrix.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(SparseMatrixTests)
	{
	public:

		TEST_METHOD(Allocation)
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

		TEST_METHOD(Copy)
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

			Assert::IsTrue(m1 == m2);

			cl::GpuDoubleSparseMatrix m3(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuDoubleSparseMatrix m4(m3);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(m3 == m4);

			cl::GpuIntegerSparseMatrix m5(4, 6, gpuNonZeroCols, gpuNonZeroRows, 10);
			dm::DeviceManager::CheckDeviceSanity();

			cl::GpuIntegerSparseMatrix m6(m5);
			dm::DeviceManager::CheckDeviceSanity();

			Assert::IsTrue(m5 == m6);
		}

		TEST_METHOD(ReadFromDense)
		{
			std::vector<float> denseMatrix(24);
			denseMatrix[10] = 2.7182f;
			denseMatrix[20] = 3.1415f;
			denseMatrix[22] = 1.6180f;

			cl::mat dv(denseMatrix, 4, 6);
			cl::smat sv(dv);

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