#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <sstream>

namespace clt
{
	class SerializationTests : public ::testing::Test
	{
	};

	/*
	*	If serialization is denoted as f(vec), and deserialization g(vec)
	*	 ==> f(g(vec)) = vec
	*/
	TEST_F(SerializationTests, SerilizationInversion)
	{
		std::stringstream s;
		cl::vec v(18u, 0.12345f);
		s << v;

		cl::vec u = cl::DeserializeVector(s);

		ASSERT_TRUE(u == v);

		std::stringstream t;
		cl::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		t << m1;

		cl::mat m2 = cl::DeserializeMatrix(t);

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6);
		}
	}
}