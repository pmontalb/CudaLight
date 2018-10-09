
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

namespace clt
{
	class CuBlasTests : public ::testing::Test
	{
	};

	static cl::mat GetInvertibleMatrix(size_t nRows, const unsigned seed = 1234)
	{
		auto A = cl::RandomUniform(nRows, nRows, seed);
		auto _A = A.Get();

		for (size_t i = 0; i < nRows; ++i)
			_A[i + nRows * i] += 2;

		A.ReadFrom(_A);
		return A;
	}

	TEST_F(CuBlasTests, Add)
	{
		cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		cl::vec v2 = cl::RandomUniform(v1.size());
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();

		auto v3 = v1 + v2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _v3 = v3.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(fabs(_v3[i] - _v1[i] - _v2[i]) <= 1e-7);

		auto v4 = v1.Add(v2, 2.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v4 = v4.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(fabs(_v4[i] - _v1[i] - 2.0 * _v2[i]) <= 1.2e-7);
	}

	TEST_F(CuBlasTests, AddMatrix)
	{
		cl::mat m1 = cl::LinSpace(-1.0f, 1.0f, 100, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		cl::mat m2 = cl::RandomUniform(m1.nRows(), m1.nCols(), 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();

		auto m3 = m1 + m2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_TRUE(fabs(_m3[i] - _m1[i] - _m2[i]) <= 1e-7);

		auto m4 = m1.Add(m2, MatrixOperation::None, MatrixOperation::None, 2.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m4 = m4.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_TRUE(fabs(_m4[i] - _m1[i] - 2.0 * _m2[i]) <= 1.2e-7);
	}

	TEST_F(CuBlasTests, Scale)
	{
		cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		v1.Scale(2.0);
		auto _v2 = v1.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(fabs(2.0 * _v1[i] - _v2[i]) <= 1e-7);
	}

	TEST_F(CuBlasTests, ElementWiseProduct)
	{
		cl::vec v1 = cl::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		cl::vec v2 = cl::RandomUniform(v1.size());
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();

		auto v3 = v1 % v2;
		auto _v3 = v3.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(fabs(_v3[i] - _v1[i] * _v2[i]) <= 1e-7);
	}

	TEST_F(CuBlasTests, Multiply)
	{
		cl::mat m1(10, 10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		cl::mat m2(10, 10, 9.8765f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();

		auto m3 = m1 * m2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			for (size_t j = 0; j < m1.nCols(); ++j)
			{
				double m1m2 = 0.0;
				for (size_t k = 0; k < m1.nCols(); ++k)
					m1m2 += _m1[i + k * m1.nRows()] * _m2[k + j * m2.nRows()];
				ASSERT_TRUE(fabs(m1m2 - _m3[i + j * m1.nRows()]) <= 5e-5);
			}
		}
	}

	TEST_F(CuBlasTests, Dot)
	{
		cl::mat m1(10, 10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		cl::vec v1(10, 9.8765f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		auto v2 = m1 * v1;
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			double m1v1 = 0.0;
			for (size_t j = 0; j < m1.nCols(); ++j)
				m1v1 += _m1[i + j * m1.nRows()] * _v1[j];
			ASSERT_TRUE(fabs(m1v1 - _v2[i]) <= 5e-5);
		}
	}

	TEST_F(CuBlasTests, Invert)
	{
		cl::mat v = GetInvertibleMatrix(128);
		dm::DeviceManager::CheckDeviceSanity();

		cl::mat vMinus1(v);
		vMinus1.Invert();
		dm::DeviceManager::CheckDeviceSanity();

		auto eye = v.Multiply(vMinus1);
		auto _eye = eye.Get();
		auto _v = v.Get();
		auto _vMinus1 = vMinus1.Get();

		for (size_t i = 0; i < v.nRows(); ++i)
		{
			for (size_t j = 0; j < v.nRows(); ++j)
			{
				double expected = i == j ? 1.0 : 0.0;
				ASSERT_TRUE(fabs(_eye[i + v.nRows() * j] - expected) <= 5e-5);
			}
		}
	}

	TEST_F(CuBlasTests, Solve)
	{
		cl::mat v = GetInvertibleMatrix(128);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();

		cl::mat u = GetInvertibleMatrix(v.nRows(), 2345);
		auto _u = u.Get();
		v.Solve(u);
		dm::DeviceManager::CheckDeviceSanity();
		auto _x = u.Get();

		auto uSanity = v.Multiply(u);
		auto _uSanity = uSanity.Get();

		for (size_t i = 0; i < v.nRows(); ++i)
		{
			for (size_t j = 0; j < v.nRows(); ++j)
			{
				double expected = _u[i + v.nRows() * j];
				ASSERT_TRUE(fabs(_uSanity[i + v.nRows() * j] - expected) <= 5e-5);
			}
		}
	}

	TEST_F(CuBlasTests, KroneckerProduct)
	{
		cl::vec u(64, 0.1);
		dm::DeviceManager::CheckDeviceSanity();
		auto _u = u.Get();

		cl::vec v(128, 0.2);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();

		cl::mat A = cl::mat::KroneckerProduct(u, v, 2.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _A = A.Get();
		ASSERT_EQ(A.nRows(), u.size());
		ASSERT_EQ(A.nCols(), v.size());

		for (size_t i = 0; i < A.nRows(); ++i)
		{
			for (size_t j = 0; j < A.nCols(); ++j)
			{
				double expected = 2.0 * _u[i] * _v[j];
				ASSERT_TRUE(fabs(_A[i + A.nRows() * j] - expected) <= 5e-5);
			}
		}
	}

	TEST_F(CuBlasTests, AbsoluteMinMax)
	{
		cl::vec x = cl::LinSpace(-1.0f, 1.0f, 128);
		auto _x = x.Get();

		float xMin = x.MinimumInAbsoluteValue();
		float xMax = x.MaximumInAbsoluteValue();
		
		float _min = 1e9, _max = 0.0;
		for (size_t i = 0; i < x.size(); i++)
		{
			if (fabs(_x[i]) < fabs(_min))
				_min = _x[i];
			if (fabs(_x[i]) > fabs(_max))
				_max = _x[i];
		}

		ASSERT_TRUE(fabs(_min - xMin) <= 1e-7);
		ASSERT_TRUE(fabs(_max - xMax) <= 1e-7);
	}
}