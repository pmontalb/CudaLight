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
		cl::mat v = cl::Eye(128);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (size_t i = 0; i < v.nRows(); ++i)
			ASSERT_TRUE(fabs(_v[i + v.nRows() * i] - 1.0) <= 5e-16);
	}

	TEST_F(MatrixTests, Linspace)
	{
		cl::mat v = cl::LinSpace(0.0f, 1.0f, 10, 10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		ASSERT_TRUE(fabs(_v[0] - 0.0) <= 1e-7);
		ASSERT_TRUE(fabs(_v[_v.size() - 1] - 1.0) <= 1e-7);
	}

	TEST_F(MatrixTests, RandomUniform)
	{
		cl::mat v = cl::RandomUniform(10, 10, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		for (const auto& iter : _v)
			ASSERT_TRUE(iter >= 0.0 && iter <= 1.0);
	}

	TEST_F(MatrixTests, RandomGaussian)
	{
		cl::mat v = cl::RandomGaussian(10, 10, 1234);
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
}