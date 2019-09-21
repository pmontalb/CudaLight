
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>

namespace clt
{
	class CuBlasTests : public ::testing::Test
	{
	};

	static cl::mat GetInvertibleMatrix(size_t nRows, const unsigned seed = 1234)
	{
		cl::mat A = cl::mat::RandomUniform(nRows, nRows, seed);
		auto _A = A.Get();

		for (size_t i = 0; i < nRows; ++i)
			_A[i + nRows * i] += 2;

		A.ReadFrom(_A);
		return A;
	}

	TEST_F(CuBlasTests, Add)
	{
		cl::vec v1 = cl::vec::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		cl::vec v2 = cl::vec::RandomUniform(v1.size(), 1234);
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

		cl::ivec v5(32, 5);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v5 = v5.Get();

		cl::ivec v6(32, 7);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v6 = v6.Get();

		auto v7 = v5.Add(v6, 3);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v7 = v7.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v7[i], _v5[i] + 3 * _v6[i]);

		auto v8 = v5.Add(v6, -2);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v8 = v8.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v8[i], _v5[i] - 2 * _v6[i]);

		v5.AddEqual(v6, +10);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v5New = v5.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v5New[i], _v5[i] + 10 * _v6[i]);
	}

	TEST_F(CuBlasTests, AddMatrix)
	{
		cl::mat m1 = cl::mat::LinSpace(-1.0f, 1.0f, 100, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		cl::mat m2 = cl::mat::RandomUniform(m1.nRows(), m1.nCols(), 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();

		auto m3 = m1 + m2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_TRUE(fabs(_m3[i] - _m1[i] - _m2[i]) <= 1e-7);

		auto m4 = m1.Add(m2, MatrixOperation::None, MatrixOperation::None, 2.0, 3.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m4 = m4.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_LT(fabs(_m4[i] / (2.0 * _m1[i] + 3.0 * _m2[i]) - 1.0), 1e-7) << i << "; " << _m4[i] << "; " << 2.0 * _m1[i] - 3.0 * _m2[i];
	}
	
	TEST_F(CuBlasTests, BroadcastAdd)
	{
		cl::mat m1 = cl::mat::LinSpace(-1.0f, 1.0f, 64, 128);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();
		auto m1Copy = m1;
		
		cl::vec v1 = cl::vec::RandomUniform(m1.nRows(), 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();
		
		cl::vec v2 = cl::vec::RandomUniform(m1.nCols(), 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();
		
		auto m2 = m1.AddEqual(v1, false, 2.5);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();
		
		auto m3 = m1Copy.AddEqual(v2, true, 5.2);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();
		
		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			for (size_t j = 0; j < m1.nCols(); ++j)
			{
				ASSERT_NEAR(_m2[i + j * m1.nRows()], _m1[i + j * m1.nRows()] + 2.5 * _v1[i], 5e-7);
				ASSERT_NEAR(_m3[i + j * m1.nRows()], _m1[i + j * m1.nRows()] + 5.2 * _v2[j], 5e-7);
			}
		}
	}
	
	TEST_F(CuBlasTests, Reciprocal)
	{
		cl::vec v1 = cl::vec::LinSpace(1.0, 2.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();
		
		v1.Reciprocal();
		auto _v2 = v1.Get();
		
		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_NEAR(1.0 / _v1[i], _v2[i], 1e-7);
	}
	
	TEST_F(CuBlasTests, Scale)
	{
		cl::vec v1 = cl::vec::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		v1.Scale(2.0);
		auto _v2 = v1.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(fabs(2.0 * _v1[i] - _v2[i]) <= 1e-7);
	}
	
	TEST_F(CuBlasTests, ScaleColumns)
	{
		cl::mat m = cl::mat::RandomUniform(10, 100, 1234);
		cl::vec v(m.nCols(), 2.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		auto _m1 = m.Get();
		
		m.ScaleColumns(v);
		auto _m2 = m.Get();
		
		for (size_t i = 0; i < m.nRows(); ++i)
		{
			for (size_t j = 0; j < m.nCols(); ++j)
			ASSERT_NEAR(_m2[i + j * m.nRows()], _m1[i + j * m.nRows()] * _v[j], 1e-7);
		}
	}
	
	TEST_F(CuBlasTests, ElementWiseProduct)
	{
		cl::vec v1 = cl::vec::LinSpace(-1.0, 1.0, 100);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		cl::vec v2 = cl::vec::RandomUniform(v1.size(), 1234);
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
	
	TEST_F(CuBlasTests, SubMultiply)
	{
		cl::mat m1(10, 10, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();
		
		cl::mat m2(10, 10, 9.8765f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();
		
		cl::mat m3(m1.nRows(), m2.nCols(), -123456789);
		auto _initialM3 = m3.Get();
		
		cl::mat m4 = m1 * m2;
		auto _m4 = m4.Get();
		
		const size_t rowStartM1 = 2;
		const size_t nRowsM1 = 3;
		
		const size_t colStartM1 = 4;
		const size_t nColsM1 = 4;
		
		const size_t rowStartM2 = 3;
		const size_t colStartM2 = 3;
		const size_t nColsM2 = 5;
		m1.SubMultiply(m3, m2, rowStartM1, colStartM1, nRowsM1, nColsM1, colStartM2, nColsM2);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();
		
		for (size_t i = rowStartM1; i < rowStartM1 + nRowsM1; ++i)
		{
			for (size_t j = colStartM2; j < colStartM2 + nColsM2; ++j)
			{
				double m1m2 = 0.0;
				for (size_t k = 0; k < nColsM1; ++k)
					m1m2 += _m1[i + (k + colStartM1) * m1.nRows()] * _m2[(k + rowStartM2) + j * m2.nRows()];

				ASSERT_NEAR(m1m2, _m3[i + j * m1.nRows()], 5e-5) << "i=" << i << "; j=" << j << "; idx=" << i + j * m1.nRows();
			}
		}
		
		for (size_t i = 0; i < rowStartM1; ++i)
			for (size_t j = 0; j < colStartM2; ++j)
				ASSERT_NEAR(_initialM3[i + j * m1.nRows()], _m3[i + j * m1.nRows()], 5e-5);
		for (size_t i = rowStartM1 + nRowsM1; i < m1.nRows(); ++i)
			for (size_t j = colStartM2 + nColsM2; j < m1.nCols(); ++j)
				ASSERT_NEAR(_initialM3[i + j * m1.nRows()], _m3[i + j * m1.nRows()], 5e-5);
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
	
	TEST_F(CuBlasTests, RowWiseSum)
	{
		cl::mat A(128, 64);
		A.RandomGaussian(1234);
		
		const auto rowSum = A.RowWiseSum();
		const auto _A = A.Get();
		const auto _rowSum = rowSum.Get();
		
		ASSERT_EQ(rowSum.size(), A.nRows());
		for (size_t i = 0; i < A.nRows(); ++i)
		{
			double goldenRowSum = 0.0;
			for (size_t j = 0; j < A.nCols(); ++j)
				goldenRowSum += _A[i + j * A.nRows()];
			ASSERT_NEAR(goldenRowSum, _rowSum[i], 5e-6);
		}
	}
	
	TEST_F(CuBlasTests, ColumnWiseSum)
	{
		cl::mat A(128, 64);
		A.RandomGaussian(1234);
		
		const auto columnSum = A.ColumnWiseSum();
		const auto _A = A.Get();
		const auto _columnSum = columnSum.Get();
		
		ASSERT_EQ(columnSum.size(), A.nCols());
		for (size_t j = 0; j < A.nCols(); ++j)
		{
			double goldenColSum = 0.0;
			for (size_t i = 0; i < A.nRows(); ++i)
				goldenColSum += _A[i + j * A.nRows()];
			ASSERT_NEAR(goldenColSum, _columnSum[j], 5e-6);
		}
	}
	
	TEST_F(CuBlasTests, CubeWiseSum)
	{
		cl::ten T(11, 17, 29);
		dm::DeviceManager::CheckDeviceSanity();
		for (auto& matrix: T.matrices)
			matrix->RandomUniform();
		
		const auto _T = T.Get();
		dm::DeviceManager::CheckDeviceSanity();
		const auto cubeSum = T.CubeWiseSum();
		dm::DeviceManager::CheckDeviceSanity();
		
		const auto _cubeSum = cubeSum.Get();
		dm::DeviceManager::CheckDeviceSanity();
		
		ASSERT_EQ(cubeSum.nRows(), T.nRows());
		ASSERT_EQ(cubeSum.nCols(), T.nCols());
		for (size_t i = 0; i < T.nRows(); ++i)
		{
			for (size_t j = 0; j < T.nCols(); ++j)
			{
				double goldenCubeSum = 0.0;
				for (size_t k = 0; k < T.nMatrices(); ++k)
					goldenCubeSum += _T[i + j * T.nRows() + k * T.nRows() * T.nCols()];
				
				ASSERT_NEAR(goldenCubeSum / _cubeSum[i + j * T.nRows()] -1.0, 0.0,5e-7)
				   << "i=" << i << "; j=" << j << "; idx=" << i + j * T.nRows() << "; sum=" << _cubeSum[i + j * T.nRows()];
			}
		}
	}
	
	TEST_F(CuBlasTests, BatchedKroneckerProduct)
	{
		unsigned nCubes = 2;
		
		cl::mat u(3, nCubes, 1.0);
		//u.RandomUniform();
		u.columns[1]->Set(3.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _u = u.Get();
		
		cl::mat v(2, nCubes, 2.0);
		//v.RandomGaussian();
		v.columns[1]->Set(4.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();
		
		cl::ten A = cl::ten::KroneckerProduct(u, v, 1.0);
		dm::DeviceManager::CheckDeviceSanity();
		auto _A = A.Get();
		ASSERT_EQ(A.nRows(), u.nRows());
		ASSERT_EQ(A.nCols(), v.nRows());
		ASSERT_EQ(A.nMatrices(), nCubes);
		
		for (size_t k = 0; k < nCubes; ++k)
		{
			for (size_t i = 0; i < A.nRows(); ++i)
			{
				for (size_t j = 0; j < A.nCols(); ++j)
				{
					double expected = 1.0 * _u[i + k * u.nRows()] * _v[j + k * v.nRows()];
					ASSERT_NEAR(_A[i + A.nRows() * j + A.nRows() * A.nCols() * k], expected, 5e-5) << "(" << i << ", " << j << ", " << k << ")";
				}
			}
		}
	}
	
	TEST_F(CuBlasTests, ColumnWiseAbsoluteMinMax)
	{
		cl::mat A = cl::mat::LinSpace(-1.0f, 1.0f, 32, 128);
		auto _A = A.Get();

		auto AMin = A.ColumnWiseArgAbsMinimum();
		auto _AMin = AMin.Get();
		auto AMax = A.ColumnWiseArgAbsMaximum();
		auto _AMax = AMax.Get();

		std::vector<int> _min(A.nCols(), 0); 
		std::vector<int> _max(A.nCols(), 0);

		for (size_t j = 0; j < A.nCols(); j++)
		{
			for (size_t i = 0; i < A.nRows(); i++)
			{
				const size_t idx = i + A.nRows() * j;
				if (fabs(_A[idx]) < fabs(_A[_min[j] + A.nRows() * j]))
					_min[j] = i;
				if (fabs(_A[idx]) > fabs(_A[_max[j] + A.nRows() * j]))
					_max[j] = i;
			}

			ASSERT_TRUE(fabs(_min[j] - (_AMin[j] - 1)) <= 1e-7);
			ASSERT_TRUE(fabs(_max[j] - (_AMax[j] - 1)) <= 1e-7);
		}
	}

	TEST_F(CuBlasTests, CountEquals)
	{
		cl::vec u(64, 0.1);
		cl::vec v(u);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();

		ASSERT_EQ(u.CountEquals(v), u.size());
		ASSERT_EQ(v.CountEquals(u), u.size());

		_v[_v.size() / 2] *= 2;
		v.ReadFrom(_v);

		ASSERT_EQ(u.CountEquals(v), u.size() - 1);
		ASSERT_EQ(v.CountEquals(u), u.size() - 1);

		cl::vec w = cl::vec::LinSpace(-3.14, 3.14, u.size());
		ASSERT_EQ(w.CountEquals(w), w.size());
		ASSERT_EQ(w.CountEquals(u), 0);
		ASSERT_EQ(w.CountEquals(v), 0);
	}
	
	TEST_F(CuBlasTests, TransposeMultiply)
	{
		cl::mat A(64, 128);
		A.RandomUniform();
		
		cl::mat B(64, 32);  // for A^T * B
		B.RandomUniform();
		
		cl::mat C(16, 128);  // for A * C^T
		C.RandomUniform();
		
		cl::mat D(32, 64);  // for A^T * D^T
		D.RandomUniform();
		
		auto ATB  = A.Multiply(B, MatrixOperation::Transpose, MatrixOperation::None);
		auto ACT  = A.Multiply(C, MatrixOperation::None, MatrixOperation::Transpose);
		auto ATDT = A.Multiply(D, MatrixOperation::Transpose, MatrixOperation::Transpose);
		
		auto _A = A.Get();
		auto _B = B.Get();
		auto _C = C.Get();
		auto _D = D.Get();
		
		auto _ATB = ATB.Get();
		auto _ACT = ACT.Get();
		auto _ATDT = ATDT.Get();
		
		for (size_t i = 0; i < A.nRows(); ++i)
		{
			for (size_t j = 0; j < B.nCols(); ++j)
			{
				double goldenATB = 0.0;
				for (size_t k = 0; k < A.nRows(); ++k)
					goldenATB += _A[k + i * A.nRows()] * _B[k + j * B.nRows()];
				ASSERT_NEAR(goldenATB / _ATB[i + j * ATB.nRows()], 1.0, 5e-7);
			}
		}
		
		for (size_t i = 0; i < A.nRows(); ++i)
		{
			for (size_t j = 0; j < C.nRows(); ++j)
			{
				double goldenACT = 0.0;
				for (size_t k = 0; k < A.nCols(); ++k)
					goldenACT += _A[i + k * A.nRows()] * _C[j + k * C.nRows()];
				ASSERT_NEAR(goldenACT / _ACT[i + j * ACT.nRows()], 1.0, 5e-7);
			}
		}
		
		for (size_t i = 0; i < A.nCols(); ++i)
		{
			for (size_t j = 0; j < D.nRows(); ++j)
			{
				double goldenATDT = 0.0;
				for (size_t k = 0; k < A.nRows(); ++k)
					goldenATDT += _A[k + i * A.nRows()] * _D[j + k * D.nRows()];
				ASSERT_NEAR(goldenATDT / _ATDT[i + j * ATDT.nRows()], 1.0, 5e-7);
			}
		}
	}
}