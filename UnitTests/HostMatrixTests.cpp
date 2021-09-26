#include <gtest/gtest.h>

#include <ColumnWiseMatrix.h>

namespace clt
{
	class HostMatrixTests: public ::testing::Test
	{
	};

	TEST_F(HostMatrixTests, Allocation)
	{
		cl::test::mat m1(10, 5, 1.2345f);
		cl::test::dmat m2(10, 5, 1.2345);
	}

	TEST_F(HostMatrixTests, Copy)
	{
		cl::test::mat m1(10, 5, 1.2345f);
		cl::test::mat m2(m1);

		ASSERT_TRUE(m1 == m2);

		cl::test::dmat m3(10, 5, 1.2345);
		cl::test::dmat m4(m3);

		ASSERT_TRUE(m3 == m4);

		cl::test::imat m5(10, 5, 10);
		cl::test::imat m6(m5);

		ASSERT_TRUE(m5 == m6);
	}

	TEST_F(HostMatrixTests, Eye)
	{
		cl::test::mat v = cl::test::mat::Eye(128);

		auto _v = v.Get();
		for (size_t i = 0; i < v.nRows(); ++i)
			ASSERT_TRUE(std::fabs(_v[i + v.nRows() * i] - 1.0f) <= 5e-16f);
	}

	TEST_F(HostMatrixTests, Linspace)
	{
		cl::test::mat v = cl::test::mat::LinSpace(0.0f, 1.0f, 10, 10);

		auto _v = v.Get();
		ASSERT_TRUE(std::fabs(_v[0] - 0.0f) <= 1e-7f);
		ASSERT_TRUE(std::fabs(_v[_v.size() - 1] - 1.0f) <= 1e-7f);
	}

	TEST_F(HostMatrixTests, RandomUniform)
	{
		cl::test::mat v = cl::test::mat::RandomUniform(10, 10, 1234);

		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0f && iter <= 1.0f);
	}

	TEST_F(HostMatrixTests, RandomGaussian)
	{
		cl::test::mat v = cl::test::mat::RandomGaussian(10, 10, 1234);
		auto _v = v.Get();
	}

	TEST_F(HostMatrixTests, GetColumn)
	{
		cl::test::mat m1(10, 5, 1.2345f);

		for (unsigned j = 0; j < m1.nCols(); ++j)
		{
			auto col = m1.Get(j);

			ASSERT_EQ(static_cast<unsigned>(col.size()), m1.nRows());
			for (const auto elem : col)
				ASSERT_TRUE(std::fabs(elem - 1.2345f) <= 1e-7f);
		}
	}

	TEST_F(HostMatrixTests, SetColumn)
	{
		cl::test::mat m1(10, 5, 1.2345f);

		auto _m1 = m1.Get();

		const cl::test::vec v1(10, 2.3456f);

		auto _v1 = v1.Get();
		m1.Set(v1, 3);

		for (unsigned j = 0; j < m1.nCols(); ++j)
		{
			auto col = m1.Get(j);


			ASSERT_EQ(static_cast<unsigned>(col.size()), m1.nRows());

			if (j != 3)
			{
				for (size_t i = 0; i < col.size(); ++i)
					ASSERT_TRUE(std::fabs(col[i] - _m1[i + m1.nRows() * j]) <= 1e-7f);
			}
			else
			{
				for (size_t i = 0; i < col.size(); ++i)
					ASSERT_TRUE(std::fabs(col[i] - _v1[i]) <= 1e-7f);
			}
		}
	}

	TEST_F(HostMatrixTests, RandomShuffle)
	{
		cl::test::mat m = cl::test::mat::RandomGaussian(10, 20, 1234);

		auto _m1 = m.Get();

		m.RandomShuffleColumns(2345);
		auto _m2 = m.Get();

		// check columns have been permuted, not changing rows
		for (size_t j = 0; j < m.nCols(); ++j)
		{
			size_t j2 = 0;
			bool found = false;
			for (; j2 < m.nCols(); ++j2)
			{
				if (std::fabs(_m2[0 + j2 * m.nRows()] - _m1[0 + j * m.nRows()]) < 1e-12f)
				{
					found = true;
					break;
				}
			}
			ASSERT_TRUE(found);

			for (size_t i = 0; i < m.nRows(); ++i)
				ASSERT_DOUBLE_EQ(_m2[i + j2 * m.nRows()], _m1[i + j * m.nRows()]);
		}
	}

	TEST_F(HostMatrixTests, RandomShufflePair)
	{
		cl::test::mat m = cl::test::mat::RandomGaussian(10, 20, 1234);
		cl::test::mat n = cl::test::mat::RandomGaussian(15, 20, 1234);

		auto _m1 = m.Get();
		auto _n1 = n.Get();

		cl::test::mat::RandomShuffleColumnsPair(m, n, 2345);
		auto _m2 = m.Get();
		auto _n2 = n.Get();

		// check columns have been permuted, not changing rows
		for (size_t j = 0; j < m.nCols(); ++j)
		{
			size_t j2 = 0;
			bool found = false;
			for (; j2 < m.nCols(); ++j2)
			{
				if (std::fabs(_m2[0 + j2 * m.nRows()] - _m1[0 + j * m.nRows()]) < 1e-12f)
				{
					found = true;
					break;
				}
			}
			ASSERT_TRUE(found);

			size_t k2 = 0;
			found = false;
			for (; k2 < m.nCols(); ++k2)
			{
				if (std::fabs(_n2[0 + k2 * n.nRows()] - _n1[0 + j * n.nRows()]) < 1e-12f)
				{
					found = true;
					break;
				}
			}
			ASSERT_TRUE(found);
			ASSERT_EQ(k2, j2);

			for (size_t i = 0; i < m.nRows(); ++i)
			{
				ASSERT_DOUBLE_EQ(_m2[i + j2 * m.nRows()], _m1[i + j * m.nRows()]);
				ASSERT_DOUBLE_EQ(_n2[i + j2 * n.nRows()], _n1[i + j * n.nRows()]);
			}
		}
	}

	TEST_F(HostMatrixTests, SubMatrix)
	{
		cl::test::mat m = cl::test::mat::RandomGaussian(10, 20, 1234);

		const size_t nStart = 4;
		const size_t nEnd = 17;
		cl::test::mat n(m, nStart, nEnd);

		ASSERT_EQ(n.nRows(), m.nRows());
		ASSERT_EQ(n.nCols(), nEnd - nStart);

		auto _m = m.Get();
		auto _n = n.Get();

		for (size_t i = 0; i < m.nRows(); ++i)
		{
			for (size_t j = nStart; j < nEnd; ++j)
			{
				ASSERT_DOUBLE_EQ(_m[i + j * m.nRows()], _n[i + (j - nStart) * n.nRows()]);
			}
		}
	}
}	 // namespace clt
