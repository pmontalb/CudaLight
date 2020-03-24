#include <gtest/gtest.h>
#include <SparseVector.h>

namespace clt
{
	class MklSparseVectorTests : public ::testing::Test
	{
	};

	TEST_F(MklSparseVectorTests, Allocation)
	{
		std::vector<int> indices = { 0, 5 };
		cl::mkl::ivec gpuIndices(static_cast<unsigned>(indices.size()));
		gpuIndices.ReadFrom(indices);

		cl::MklSingleSparseVector v1(10, gpuIndices, 1.2345f);
		cl::MklDoubleSparseVector v2(10, gpuIndices, 1.2345);
	}

	TEST_F(MklSparseVectorTests, Copy)
	{
		std::vector<int> indices = { 0, 5 };
		cl::mkl::ivec gpuIndices(static_cast<unsigned>(indices.size()));
		gpuIndices.ReadFrom(indices);

		cl::MklSingleSparseVector v1(10, gpuIndices, 1.2345f);
		cl::MklSingleSparseVector v2(v1);

		ASSERT_TRUE(v1 == v2);

		cl::MklDoubleSparseVector v3(10, gpuIndices, 1.2345);
		cl::MklDoubleSparseVector v4(v3);

		ASSERT_TRUE(v3 == v4);

		cl::MklIntegerSparseVector v5(10, gpuIndices, 10);
		cl::MklIntegerSparseVector v6(v5);

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(MklSparseVectorTests, ReadFromDense)
	{
		std::vector<float> denseVector(50);
		denseVector[10] = 2.7182f;
		denseVector[20] = 3.1415f;
		denseVector[30] = 1.6180f;

		cl::mkl::vec dv(denseVector);
		cl::mkl::svec sv(dv);

		auto _dv = dv.Get();
		auto _sv = dv.Get();
		ASSERT_EQ(_dv.size(), _sv.size());

		for (size_t i = 0; i < _dv.size(); ++i)
		{
			ASSERT_TRUE(std::fabs(_dv[i] - _sv[i]) <= 1e-7f);
		}
	}
}
