
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>

namespace clt
{
	class MklBlasTests : public ::testing::Test
	{
	};

	static cl::mkl::mat GetInvertibleMatrix(unsigned nRows, const unsigned seed = 1234)
	{
		cl::mkl::mat A = cl::mkl::mat::RandomUniform(nRows, nRows, seed);
		auto _A = A.Get();

		for (size_t i = 0; i < nRows; ++i)
			_A[i + nRows * i] += 2;

		A.ReadFrom(_A);
		return A;
	}

	TEST_F(MklBlasTests, Add)
	{
		cl::mkl::vec v1 = cl::mkl::vec::LinSpace(-1.0, 1.0, 100);
		
		auto _v1 = v1.Get();

		cl::mkl::vec v2 = cl::mkl::vec::RandomUniform(v1.size(), 1234);
		
		auto _v2 = v2.Get();

		auto v3 = v1 + v2;
		
		auto _v3 = v3.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(std::fabs(_v3[i] - _v1[i] - _v2[i]) <= 1e-7f);

		auto v4 = v1.Add(v2, 2.0);
		
		auto _v4 = v4.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(std::fabs(_v4[i] - _v1[i] - 2.0f * _v2[i]) <= 1.2e-7f);

		cl::mkl::ivec v5(32, 5);
		
		auto _v5 = v5.Get();

		cl::mkl::ivec v6(32, 7);
		
		auto _v6 = v6.Get();

		auto v7 = v5.Add(v6, 3);
		
		auto _v7 = v7.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v7[i], _v5[i] + 3 * _v6[i]);

		auto v8 = v5.Add(v6, -2);
		
		auto _v8 = v8.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v8[i], _v5[i] - 2 * _v6[i]);

		v5.AddEqual(v6, +10);
		
		auto _v5New = v5.Get();

		for (size_t i = 0; i < v7.size(); ++i)
			ASSERT_EQ(_v5New[i], _v5[i] + 10 * _v6[i]);
	}

	TEST_F(MklBlasTests, AddMatrix)
	{
		cl::mkl::mat m1 = cl::mkl::mat::LinSpace(-1.0f, 1.0f, 100, 100);

		auto _m1 = m1.Get();

		cl::mkl::mat m2 = cl::mkl::mat::RandomUniform(m1.nRows(), m1.nCols(), 1234);

		auto _m2 = m2.Get();

		auto m3 = m1 + m2;

		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_TRUE(std::fabs(_m3[i] - _m1[i] - _m2[i]) <= 1e-7f);

		auto m4 = m1.Add(m2, MatrixOperation::None, MatrixOperation::None, 2.0, 3.0);

		auto _m4 = m4.Get();

		for (size_t i = 0; i < m1.size(); ++i)
			ASSERT_LT(std::fabs(_m4[i] / (2.0f * _m1[i] + 3.0f * _m2[i]) - 1.0f), 1e-7) << i << "; " << _m4[i] << "; " << 2.0f * _m1[i] - 3.0f * _m2[i];
	}

	TEST_F(MklBlasTests, BroadcastAdd)
	{
		cl::mkl::mat m1 = cl::mkl::mat::LinSpace(-1.0f, 1.0f, 64, 128);

		auto _m1 = m1.Get();
		auto m1Copy = m1;

		cl::mkl::vec v1 = cl::mkl::vec::RandomUniform(m1.nRows(), 1234);

		auto _v1 = v1.Get();

		cl::mkl::vec v2 = cl::mkl::vec::RandomUniform(m1.nCols(), 1234);

		auto _v2 = v2.Get();

		auto m2 = m1.AddEqualBroadcast(v1, false, 2.5);

		auto _m2 = m2.Get();

		auto m3 = m1Copy.AddEqualBroadcast(v2, true, 5.2);

		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			for (size_t j = 0; j < m1.nCols(); ++j)
			{
				ASSERT_NEAR(_m2[i + j * m1.nRows()], _m1[i + j * m1.nRows()] + 2.5f * _v1[i], 5e-7);
				ASSERT_NEAR(_m3[i + j * m1.nRows()], _m1[i + j * m1.nRows()] + 5.2f * _v2[j], 5e-7);
			}
		}
	}

	TEST_F(MklBlasTests, Reciprocal)
	{
		cl::mkl::vec v1 = cl::mkl::vec::LinSpace(1.0, 2.0, 100);
		
		auto _v1 = v1.Get();
		
		v1.Reciprocal();
		auto _v2 = v1.Get();
		
		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_NEAR(1.0f / _v1[i], _v2[i], 1e-7);
	}
	
	TEST_F(MklBlasTests, Scale)
	{
		cl::mkl::vec v1 = cl::mkl::vec::LinSpace(-1.0, 1.0, 100);
		
		auto _v1 = v1.Get();

		v1.Scale(2.0);
		auto _v2 = v1.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(std::fabs(2.0f * _v1[i] - _v2[i]) <= 1e-7f);
	}

	TEST_F(MklBlasTests, ScaleColumns)
	{
		cl::mkl::mat m = cl::mkl::mat::RandomUniform(10, 100, 1234);
		cl::mkl::vec v(m.nCols(), 2.0);

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

	TEST_F(MklBlasTests, ElementWiseProduct)
	{
		cl::mkl::vec v1 = cl::mkl::vec::LinSpace(-1.0, 1.0, 100);
		auto _v1 = v1.Get();

		cl::mkl::vec v2 = cl::mkl::vec::RandomUniform(v1.size(), 1234);
		auto _v2 = v2.Get();

		auto v3 = v1 % v2;
		v3.Print();
		auto _v3 = v3.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(std::fabs(_v3[i] - _v1[i] * _v2[i]) <= 1e-7f);
	}

	TEST_F(MklBlasTests, Multiply)
	{
		cl::mkl::mat m1(10, 10, 1.2345f);

		auto _m1 = m1.Get();

		cl::mkl::mat m2(10, 10, 9.8765f);

		auto _m2 = m2.Get();

		auto m3 = m1 * m2;

		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			for (size_t j = 0; j < m1.nCols(); ++j)
			{
				double m1m2 = 0.0;
				for (size_t k = 0; k < m1.nCols(); ++k)
					m1m2 += static_cast<double>(_m1[i + k * m1.nRows()] * _m2[k + j * m2.nRows()]);
				ASSERT_TRUE(std::fabs(static_cast<float>(m1m2) - _m3[i + j * m1.nRows()]) <= 5e-5f);
			}
		}
	}
	
	TEST_F(MklBlasTests, SubMultiply)
	{
		cl::mkl::mat m1(10, 10, 1.2345f);

		auto _m1 = m1.Get();

		cl::mkl::mat m2(10, 10, 9.8765f);

		auto _m2 = m2.Get();

		cl::mkl::mat m3(m1.nRows(), m2.nCols(), -123456789.0f);
		auto _initialM3 = m3.Get();

		cl::mkl::mat m4 = m1 * m2;
		auto _m4 = m4.Get();

		const size_t rowStartM1 = 2;
		const size_t nRowsM1 = 3;

		const size_t colStartM1 = 4;
		const size_t nColsM1 = 4;

		const size_t rowStartM2 = 3;
		const size_t colStartM2 = 3;
		const size_t nColsM2 = 5;
		m1.SubMultiply(m3, m2, rowStartM1, colStartM1, nRowsM1, nColsM1, colStartM2, nColsM2);

		auto _m3 = m3.Get();

		for (size_t i = rowStartM1; i < rowStartM1 + nRowsM1; ++i)
		{
			for (size_t j = colStartM2; j < colStartM2 + nColsM2; ++j)
			{
				double m1m2 = 0.0;
				for (size_t k = 0; k < nColsM1; ++k)
					m1m2 += static_cast<double>(_m1[i + (k + colStartM1) * m1.nRows()] * _m2[(k + rowStartM2) + j * m2.nRows()]);

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

	TEST_F(MklBlasTests, Dot)
	{
		cl::mkl::mat m1(10, 10, 1.2345f);

		auto _m1 = m1.Get();

		cl::mkl::vec v1(10, 9.8765f);

		auto _v1 = v1.Get();

		auto v2 = m1 * v1;

		auto _v2 = v2.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			double m1v1 = 0.0;
			for (size_t j = 0; j < m1.nCols(); ++j)
				m1v1 += static_cast<double>(_m1[i + j * m1.nRows()] * _v1[j]);
			ASSERT_TRUE(std::fabs(static_cast<float>(m1v1) - _v2[i]) <= 5e-5f);
		}
	}

	TEST_F(MklBlasTests, Invert)
	{
		cl::mkl::mat v = GetInvertibleMatrix(128);

		cl::mkl::mat vMinus1(v);
		vMinus1.Invert();

		auto eye = v.Multiply(vMinus1);
		auto _eye = eye.Get();
		auto _v = v.Get();
		auto _vMinus1 = vMinus1.Get();

		for (size_t i = 0; i < v.nRows(); ++i)
		{
			for (size_t j = 0; j < v.nRows(); ++j)
			{
				float expected = i == j ? 1.0 : 0.0;
				ASSERT_TRUE(std::fabs(_eye[i + v.nRows() * j] - expected) <= 5e-5f);
			}
		}
	}

	TEST_F(MklBlasTests, Solve)
	{
		cl::mkl::mat v = GetInvertibleMatrix(128);

		auto _v = v.Get();

		cl::mkl::mat u = GetInvertibleMatrix(v.nRows(), 2345);
		auto _u = u.Get();
		v.Solve(u);

		auto _x = u.Get();

		auto uSanity = v.Multiply(u);
		auto _uSanity = uSanity.Get();

		for (size_t i = 0; i < v.nRows(); ++i)
		{
			for (size_t j = 0; j < v.nRows(); ++j)
			{
				float expected = _u[i + v.nRows() * j];
				ASSERT_TRUE(std::fabs(_uSanity[i + v.nRows() * j] - expected) <= 5e-5f);
			}
		}
	}

	TEST_F(MklBlasTests, KroneckerProduct)
	{
		cl::mkl::vec u(64, 0.1f);

		auto _u = u.Get();

		cl::mkl::vec v(128, 0.2f);

		auto _v = v.Get();

		cl::mkl::mat A = cl::mkl::mat::KroneckerProduct(u, v, 2.0);

		auto _A = A.Get();
		ASSERT_EQ(A.nRows(), u.size());
		ASSERT_EQ(A.nCols(), v.size());

		for (size_t i = 0; i < A.nRows(); ++i)
		{
			for (size_t j = 0; j < A.nCols(); ++j)
			{
				float expected = 2.0f * _u[i] * _v[j];
				ASSERT_TRUE(std::fabs(_A[i + A.nRows() * j] - expected) <= 5e-5f);
			}
		}
	}

	TEST_F(MklBlasTests, RowWiseSum)
	{
		cl::mkl::mat A(128, 64);
		A.RandomGaussian(1234);

		const auto rowSum = A.RowWiseSum();
		const auto _A = A.Get();
		const auto _rowSum = rowSum.Get();

		ASSERT_EQ(rowSum.size(), A.nRows());
		for (size_t i = 0; i < A.nRows(); ++i)
		{
			double goldenRowSum = 0.0;
			for (size_t j = 0; j < A.nCols(); ++j)
				goldenRowSum += static_cast<double>(_A[i + j * A.nRows()]);
			ASSERT_NEAR(goldenRowSum, _rowSum[i], 5e-6);
		}
	}

	TEST_F(MklBlasTests, ColumnWiseSum)
	{
		cl::mkl::mat A(128, 64);
		A.RandomGaussian(1234);

		const auto columnSum = A.ColumnWiseSum();
		const auto _A = A.Get();
		const auto _columnSum = columnSum.Get();

		ASSERT_EQ(columnSum.size(), A.nCols());
		for (size_t j = 0; j < A.nCols(); ++j)
		{
			double goldenColSum = 0.0;
			for (size_t i = 0; i < A.nRows(); ++i)
				goldenColSum += static_cast<double>(_A[i + j * A.nRows()]);
			ASSERT_NEAR(goldenColSum, _columnSum[j], 5e-6);
		}
	}

	TEST_F(MklBlasTests, CubeWiseSum)
	{
		cl::mkl::ten T(64, 128, 32);

		double x = 0.0;
		for (auto& matrix: T.matrices)
			matrix->Set(static_cast<float>(++x));

		const auto _T = T.Get();

		const auto cubeSum = T.CubeWiseSum();
		const auto _cubeSum = cubeSum.Get();

		ASSERT_EQ(cubeSum.nRows(), T.nRows());
		ASSERT_EQ(cubeSum.nCols(), T.nCols());
		for (size_t i = 0; i < T.nRows(); ++i)
		{
			for (size_t j = 0; j < T.nCols(); ++j)
			{
				double goldenCubeSum = 0.0;
				for (size_t k = 0; k < T.nMatrices(); ++k)
					goldenCubeSum += static_cast<double>(_T[i + j * T.nRows() + k * T.nRows() * T.nCols()]);

				ASSERT_NEAR(goldenCubeSum / static_cast<double>(_cubeSum[i + j * T.nRows()]) - 1.0, 0.0,5e-7)
				   << "i=" << i << "; j=" << j << "; idx=" << i + j * T.nRows() << "; sum=" << _cubeSum[i + j * T.nRows()];
			}
		}

		for (size_t n = 0; n < 10; ++n)
		{
			cl::mkl::mat out(T.nRows(), T.nCols(), 0.0);
			T.CubeWiseSum(out);
			const auto _cubeSum1 = out.Get();


			ASSERT_EQ(out.nRows(), T.nRows());
			ASSERT_EQ(out.nCols(), T.nCols());
			for (size_t i = 0; i < T.nRows(); ++i)
			{
				for (size_t j = 0; j < T.nCols(); ++j)
				{
					double goldenCubeSum = 0.0;
					for (size_t k = 0; k < T.nMatrices(); ++k)
						goldenCubeSum += static_cast<double>(_T[i + j * T.nRows() + k * T.nRows() * T.nCols()]);

					ASSERT_NEAR(goldenCubeSum / static_cast<double>(_cubeSum1[i + j * T.nRows()]) - 1.0, 0.0,5e-7)
												<< "i=" << i << "; j=" << j << "; idx=" << i + j * T.nRows() << "; sum=" << _cubeSum1[i + j * T.nRows()] << ";n=" << n;
				}
			}

		}
	}

	TEST_F(MklBlasTests, BatchedKroneckerProduct)
	{
		unsigned nCubes = 64;

		cl::mkl::mat u(128, nCubes, 1.0);
		u.RandomUniform();

		auto _u = u.Get();

		cl::mkl::mat v(32, nCubes, 2.0);
		v.RandomGaussian();

		auto _v = v.Get();

		cl::mkl::ten A = cl::mkl::ten::KroneckerProduct(u, v, 1.0);

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
					float expected = 1.0f * _u[i + k * u.nRows()] * _v[j + k * v.nRows()];
					ASSERT_NEAR(_A[i + A.nRows() * j + A.nRows() * A.nCols() * k], expected, 5e-5) << "(" << i << ", " << j << ", " << k << ")";
				}
			}
		}
	}

	TEST_F(MklBlasTests, ColumnWiseAbsoluteMinMax)
	{
		cl::mkl::mat A = cl::mkl::mat::LinSpace(-1.0f, 1.0f, 32, 128);
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
				if (std::fabs(_A[idx]) < std::fabs(_A[static_cast<size_t>(_min[j]) + A.nRows() * j]))
					_min[j] = static_cast<int>(i);
				if (std::fabs(_A[idx]) > std::fabs(_A[static_cast<size_t>(_max[j]) + A.nRows() * j]))
					_max[j] = static_cast<int>(i);
			}
		}

		for (size_t j = 0; j < A.nCols(); j++)
		{
			ASSERT_TRUE(std::fabs(_min[j] - (_AMin[j] - 1)) <= 1e-7);
			ASSERT_TRUE(std::fabs(_max[j] - (_AMax[j] - 1)) <= 1e-7);
		}
	}

	TEST_F(MklBlasTests, CountEquals)
	{
		cl::mkl::vec u(64, 0.1f);
		cl::mkl::vec v(u);
		
		auto _v = v.Get();

		ASSERT_EQ(u.CountEquals(v), u.size());
		ASSERT_EQ(v.CountEquals(u), u.size());

		_v[_v.size() / 2] *= 2;
		v.ReadFrom(_v);

		ASSERT_EQ(u.CountEquals(v), u.size() - 1);
		ASSERT_EQ(v.CountEquals(u), u.size() - 1);

		cl::mkl::vec w = cl::mkl::vec::LinSpace(-3.14f, 3.14f, u.size());
		ASSERT_EQ(w.CountEquals(w), w.size());
		ASSERT_EQ(w.CountEquals(u), 0);
		ASSERT_EQ(w.CountEquals(v), 0);
	}
	
	TEST_F(MklBlasTests, TransposeMultiply)
	{
		cl::mkl::mat A(64, 128);
		A.RandomUniform();

		cl::mkl::mat B(64, 32);  // for A^T * B
		B.RandomUniform();

		cl::mkl::mat C(16, 128);  // for A * C^T
		C.RandomUniform();

		cl::mkl::mat D(32, 64);  // for A^T * D^T
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
					goldenATB += static_cast<double>(_A[k + i * A.nRows()] * _B[k + j * B.nRows()]);
				ASSERT_NEAR(goldenATB / static_cast<double>(_ATB[i + j * ATB.nRows()]), 1.0, 5e-7);
			}
		}

		for (size_t i = 0; i < A.nRows(); ++i)
		{
			for (size_t j = 0; j < C.nRows(); ++j)
			{
				double goldenACT = 0.0;
				for (size_t k = 0; k < A.nCols(); ++k)
					goldenACT += static_cast<double>(_A[i + k * A.nRows()] * _C[j + k * C.nRows()]);
				ASSERT_NEAR(goldenACT / static_cast<double>(_ACT[i + j * ACT.nRows()]), 1.0, 6e-7);
			}
		}

		for (size_t i = 0; i < A.nCols(); ++i)
		{
			for (size_t j = 0; j < D.nRows(); ++j)
			{
				double goldenATDT = 0.0;
				for (size_t k = 0; k < A.nRows(); ++k)
					goldenATDT += static_cast<double>(_A[k + i * A.nRows()] * _D[j + k * D.nRows()]);
				ASSERT_NEAR(goldenATDT / static_cast<double>(_ATDT[i + j * ATDT.nRows()]), 1.0, 5e-7);
			}
		}
	}
}
