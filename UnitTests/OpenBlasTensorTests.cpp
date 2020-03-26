#include <gtest/gtest.h>

#include <Tensor.h>

namespace clt
{
	class OpenBlasTensorTests : public ::testing::Test
	{
	};

	TEST_F(OpenBlasTensorTests, Allocation)
	{
		cl::oblas::ten t1(10, 5, 5, 1.2345f);
		cl::oblas::dten t2(10, 5, 5, 1.2345);
	}

	TEST_F(OpenBlasTensorTests, Copy)
	{
		cl::oblas::ten t1(10, 5, 5, 1.2345f);
		cl::oblas::ten t2(t1);
		
		ASSERT_TRUE(t1 == t2);

		cl::oblas::dten t3(10, 5, 5, 1.2345);
		cl::oblas::dten t4(t3);
		
		ASSERT_TRUE(t3 == t4);

		cl::oblas::iten v5(10, 5, 5, 10);
		cl::oblas::iten v6(v5);

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(OpenBlasTensorTests, Linspace)
	{
		cl::oblas::ten v = cl::oblas::ten::LinSpace(0.0f, 1.0f, 10, 10, 10);
		
		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(OpenBlasTensorTests, RandomUniform)
	{
		cl::oblas::ten v = cl::oblas::ten::RandomUniform(10, 10, 10, 1234);
		
		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(OpenBlasTensorTests, RandomGaussian)
	{
		cl::oblas::ten v = cl::oblas::ten::RandomGaussian(10, 10, 10, 1234);
	}
}
