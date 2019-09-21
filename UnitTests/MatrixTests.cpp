#include <gtest/gtest.h>

#include <ColumnWiseMatrix.h>

namespace clt
{
	class MatrixTests : public ::testing::Test
	{
	};

	TEST_F(MatrixTests, Allocation)
	{
		cl::GpuSingleMatrix m1(10, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleMatrix m2(10, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuSingleMatrix m3(10, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::CpuDoubleMatrix m4(10, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();
	}

	TEST_F(MatrixTests, Copy)
	{
		cl::GpuSingleMatrix m1(10, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuSingleMatrix m2(m1);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m1 == m2);

		cl::GpuDoubleMatrix m3(10, 5, 1.2345);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuDoubleMatrix m4(m3);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m3 == m4);

		cl::GpuIntegerMatrix m5(10, 5, 10);
		dm::DeviceManager::CheckDeviceSanity();

		cl::GpuIntegerMatrix m6(m5);
		dm::DeviceManager::CheckDeviceSanity();

		ASSERT_TRUE(m5 == m6);
	}

	TEST_F(MatrixTests, Eye)
	{
		cl::mat v = cl::mat::Eye(128);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < v.nRows(); ++i)
			ASSERT_TRUE(fabs(_v[i + v.nRows() * i] - 1.0) <= 5e-16);
	}

	TEST_F(MatrixTests, Linspace)
	{
		cl::mat v = cl::mat::LinSpace(0.0f, 1.0f, 10, 10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		ASSERT_TRUE(fabs(_v[0] - 0.0) <= 1e-7);
		ASSERT_TRUE(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
	}

	TEST_F(MatrixTests, RandomUniform)
	{
		cl::mat v = cl::mat::RandomUniform(10, 10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0 && iter <= 1.0);
	}

	TEST_F(MatrixTests, RandomGaussian)
	{
		cl::mat v = cl::mat::RandomGaussian(10, 10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < _v.size() / 2; ++i)
			ASSERT_TRUE(fabs(_v[2 * i] + _v[2 * i + 1]) <= 1e-7);
	}

	TEST_F(MatrixTests, GetColumn)
	{
		cl::mat m1(10, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();

		for (unsigned j = 0; j < m1.nCols(); ++j)
		{
			auto col = m1.Get(j);
			dm::DeviceManager::CheckDeviceSanity();
				
			ASSERT_EQ(static_cast<unsigned>(col.size()), m1.nRows());
			for (size_t i = 0; i < col.size(); ++i)
				ASSERT_TRUE(fabs(col[i] - 1.2345) <= 1e-7);
		}

	}

	TEST_F(MatrixTests, SetColumn)
	{
		cl::mat m1(10, 5, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		const cl::vec v1(10, 2.3456f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();
		m1.Set(v1, 3);

		for (unsigned j = 0; j < m1.nCols(); ++j)
		{
			auto col = m1.Get(j);
			dm::DeviceManager::CheckDeviceSanity();

			ASSERT_EQ(static_cast<unsigned>(col.size()), m1.nRows());

			if (j != 3)
			{
				for (size_t i = 0; i < col.size(); ++i)
					ASSERT_TRUE(fabs(col[i] - _m1[i + m1.nRows() * j]) <= 1e-7);
			}
			else
			{
				for (size_t i = 0; i < col.size(); ++i)
					ASSERT_TRUE(fabs(col[i] - _v1[i]) <= 1e-7);
			}
		}
	}
	
	TEST_F(MatrixTests, RandomShuffle)
	{
		cl::mat m = cl::mat::RandomGaussian(10, 20, 1234);
		dm::DeviceManager::CheckDeviceSanity();
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
				if (fabs(_m2[0 + j2 * m.nRows()] - _m1[0 + j * m.nRows()]) < 1e-12)
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
	
	TEST_F(MatrixTests, RandomShufflePair)
	{
		cl::mat m = cl::mat::RandomGaussian(10, 20, 1234);
		cl::mat n = cl::mat::RandomGaussian(15, 20, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m.Get();
		auto _n1 = n.Get();
		
		cl::mat::RandomShuffleColumnsPair(m, n, 2345);
		auto _m2 = m.Get();
		auto _n2 = n.Get();
		
		// check columns have been permuted, not changing rows
		for (size_t j = 0; j < m.nCols(); ++j)
		{
			size_t j2 = 0;
			bool found = false;
			for (; j2 < m.nCols(); ++j2)
			{
				if (fabs(_m2[0 + j2 * m.nRows()] - _m1[0 + j * m.nRows()]) < 1e-12)
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
				if (fabs(_n2[0 + k2 * n.nRows()] - _n1[0 + j * n.nRows()]) < 1e-12)
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
	
	TEST_F(MatrixTests, SubMatrix)
	{
		cl::mat m = cl::mat::RandomGaussian(10, 20, 1234);
		
		const size_t nStart = 4;
		const size_t nEnd = 17;
		cl::mat n(m, nStart, nEnd);
		
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
}