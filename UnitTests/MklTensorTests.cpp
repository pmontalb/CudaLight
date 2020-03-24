#include <gtest/gtest.h>

#include <Tensor.h>

namespace clt
{
	class MklTensorTests : public ::testing::Test
	{
	};

	TEST_F(MklTensorTests, Allocation)
	{
		cl::mkl::ten t1(10, 5, 5, 1.2345f);
		cl::mkl::dten t2(10, 5, 5, 1.2345);
	}

	TEST_F(MklTensorTests, Copy)
	{
		cl::mkl::ten t1(10, 5, 5, 1.2345f);
		cl::mkl::ten t2(t1);
		
		ASSERT_TRUE(t1 == t2);

		cl::mkl::dten t3(10, 5, 5, 1.2345);
		cl::mkl::dten t4(t3);
		
		ASSERT_TRUE(t3 == t4);

		cl::mkl::iten v5(10, 5, 5, 10);
		cl::mkl::iten v6(v5);

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(MklTensorTests, Linspace)
	{
		cl::mkl::ten v = cl::mkl::ten::LinSpace(0.0f, 1.0f, 10, 10, 10);
		
		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(MklTensorTests, RandomUniform)
	{
		cl::mkl::ten v = cl::mkl::ten::RandomUniform(10, 10, 10, 1234);
		
		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(MklTensorTests, RandomGaussian)
	{
		cl::mkl::ten v = cl::mkl::ten::RandomGaussian(10, 10, 10, 1234);
	}
}
