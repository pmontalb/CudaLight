#include <gtest/gtest.h>

#include <Tensor.h>

namespace clt
{
	class GenericBlasTensorTests: public ::testing::Test
	{
	};

	TEST_F(GenericBlasTensorTests, Allocation)
	{
		cl::gblas::ten t1(10, 5, 5, 1.2345f);
		cl::gblas::dten t2(10, 5, 5, 1.2345);
	}

	TEST_F(GenericBlasTensorTests, Copy)
	{
		cl::gblas::ten t1(10, 5, 5, 1.2345f);
		cl::gblas::ten t2(t1);

		ASSERT_TRUE(t1 == t2);

		cl::gblas::dten t3(10, 5, 5, 1.2345);
		cl::gblas::dten t4(t3);

		ASSERT_TRUE(t3 == t4);

		cl::gblas::iten v5(10, 5, 5, 10);
		cl::gblas::iten v6(v5);

		ASSERT_TRUE(v5 == v6);
	}

	TEST_F(GenericBlasTensorTests, Linspace)
	{
		cl::gblas::ten v = cl::gblas::ten::LinSpace(0.0f, 1.0f, 10, 10, 10);

		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(GenericBlasTensorTests, RandomUniform)
	{
		cl::gblas::ten v = cl::gblas::ten::RandomUniform(10, 10, 10, 1234);

		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(GenericBlasTensorTests, RandomGaussian) { cl::gblas::ten v = cl::gblas::ten::RandomGaussian(10, 10, 10, 1234); }
}	 // namespace clt
