#include <gtest/gtest.h>

#include <Tensor.h>

namespace clt
{
	class HostTensorTests: public ::testing::Test
	{
	};

	TEST_F(HostTensorTests, Allocation)
	{
		cl::test::ten t1(10, 5, 5, 1.2345f);
		cl::test::dten t2(10, 5, 5, 1.2345);
	}

	TEST_F(HostTensorTests, Copy)
	{
		cl::test::ten t1(10, 5, 5, 1.2345f);
		cl::test::ten t2(t1);

		ASSERT_TRUE(t1 == t2);

		cl::test::dten t3(10, 5, 5, 1.2345);
		cl::test::dten t4(t3);

		ASSERT_TRUE(t3 == t4);

		cl::test::iten v5(10, 5, 5, 10);
		cl::test::iten v6(v5);

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(HostTensorTests, Linspace)
	{
		cl::ten v = cl::ten::LinSpace(0.0f, 1.0f, 10, 10, 10);

		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(HostTensorTests, RandomUniform)
	{
		cl::ten v = cl::ten::RandomUniform(10, 10, 10, 1234);

		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(HostTensorTests, RandomGaussian)
	{
		cl::ten v = cl::ten::RandomGaussian(10, 10, 10, 1234);

		auto _v = v.Get();
		for (size_t i = 0; i < _v.size() / 2; ++i)
			ASSERT_TRUE(std::fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7f);
	}
}	 // namespace clt
