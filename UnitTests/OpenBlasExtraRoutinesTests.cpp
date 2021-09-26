#include <gtest/gtest.h>

#include <Vector.h>
#include <cmath>

namespace clt
{
	class OpenBlasExtraRoutinesTests: public ::testing::Test
	{
	};

	TEST_F(OpenBlasExtraRoutinesTests, Sum)
	{
		const size_t size = 1024;
		auto v1 = cl::oblas::dvec::LinSpace(-1.0, 1.0, size);
		auto _v1 = v1.Get();

		double sum = v1.Sum();
		double sumSanity = 0.0;
		for (size_t i = 0; i < v1.size(); ++i)
			sumSanity += _v1[i];

		ASSERT_NEAR(sumSanity, sum, 1e-12);

		auto v2 = cl::oblas::vec::LinSpace(-1.0, 1.0, size + 1);
		auto _v2 = v2.Get();

		float sum2 = v2.Sum();
		float sumSanity2 = 0.0;
		for (size_t i = 0; i < v2.size(); ++i)
			sumSanity2 += _v2[i];

		ASSERT_NEAR(sumSanity2, sum2, 1e-7);

		cl::oblas::ivec v3(size, 1);
		auto _v3 = v3.Get();

		int sum3 = v3.Sum();
		int sumSanity3 = 0.0;
		for (size_t i = 0; i < v3.size(); ++i)
			sumSanity3 += _v3[i];

		ASSERT_NEAR(sumSanity3, sum3, 0);
	}

	TEST_F(OpenBlasExtraRoutinesTests, AbsoluteMinMax)
	{
		auto x = cl::oblas::vec::LinSpace(-1.0f, 1.0f, 128);
		auto _x = x.Get();

		float xMin = x.AbsoluteMinimum();
		float xMax = x.AbsoluteMaximum();

		float _min = 1e9, _max = 0.0;
		for (size_t i = 0; i < x.size(); i++)
		{
			if (std::fabs(_x[i]) < std::fabs(_min))
				_min = std::fabs(_x[i]);
			if (std::fabs(_x[i]) > std::fabs(_max))
				_max = std::fabs(_x[i]);
		}

		ASSERT_TRUE(std::fabs(_min - xMin) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_max - xMax) <= 1e-7f);
	}

	TEST_F(OpenBlasExtraRoutinesTests, MinMax)
	{
		auto x = cl::oblas::vec::LinSpace(-1.0f, 1.0f, 128);
		auto _x = x.Get();

		float xMin = x.Minimum();
		float xMax = x.Maximum();

		float _min = 1e9, _max = 0.0;
		for (size_t i = 0; i < x.size(); i++)
		{
			if (std::fabs(_x[i]) < _min)
				_min = _x[i];
			if (std::fabs(_x[i]) > _max)
				_max = _x[i];
		}

		ASSERT_TRUE(std::fabs(_min - xMin) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_max - xMax) <= 1e-7f);
	}
}	 // namespace clt
