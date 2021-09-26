#include <gtest/gtest.h>

#include <CompressedSparseRowMatrix.h>

namespace clt
{
	class MklSparseMatrixTests: public ::testing::Test
	{
	};

	TEST_F(MklSparseMatrixTests, Allocation)
	{
		std::vector<int> _NonZeroCols = { 0, 1, 1, 3, 2, 3, 4, 5 };
		mkl::ivec gpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		gpuNonZeroCols.ReadFrom(_NonZeroCols);

		std::vector<int> _NonZeroRows = { 0, 2, 4, 7, 8 };
		mkl::ivec gpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		gpuNonZeroRows.ReadFrom(_NonZeroRows);

		mkl::smat m1(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345f);
		mkl::dsmat m2(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345);
	}

	TEST_F(MklSparseMatrixTests, Copy)
	{
		std::vector<int> _NonZeroCols = { 0, 1, 1, 3, 2, 3, 4, 5 };
		mkl::ivec gpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		gpuNonZeroCols.ReadFrom(_NonZeroCols);

		std::vector<int> _NonZeroRows = { 0, 2, 4, 7, 8 };
		mkl::ivec gpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		gpuNonZeroRows.ReadFrom(_NonZeroRows);

		mkl::smat m1(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345f);
		mkl::smat m2(m1);

		ASSERT_TRUE(m1 == m2);

		mkl::dsmat m3(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345);
		mkl::dsmat m4(m3);

		ASSERT_TRUE(m3 == m4);
	}

	TEST_F(MklSparseMatrixTests, ReadFromDense)
	{
		std::vector<float> denseMatrix(24);
		denseMatrix[10] = 2.7182f;
		denseMatrix[20] = 3.1415f;
		denseMatrix[22] = 1.6180f;

		mkl::mat dv(denseMatrix, 4, 6);
		mkl::smat sv(dv);

		auto _dv = dv.Get();
		auto _sv = dv.Get();
		ASSERT_EQ(_dv.size(), _sv.size());

		for (size_t i = 0; i < _dv.size(); ++i)
		{
			ASSERT_TRUE(std::fabs(_dv[i] - _sv[i]) <= 1e-7f);
		}
	}
}	 // namespace clt
