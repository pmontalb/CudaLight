#include <gtest/gtest.h>

#include <CompressedSparseRowMatrix.h>
#include <SparseVector.h>

namespace clt
{
	class CuSparseTests: public ::testing::Test
	{
	};

	static cl::smat GetInvertibleSparseMatrix(unsigned nRows)
	{
		cl::mat A(nRows, nRows, 0.0);
		auto _A = A.Get();

		for (size_t i = 0; i < nRows; ++i)
		{
			_A[i + nRows * i] += 2.0f;
			if (i < nRows - 1)
				_A[i + nRows * (i + 1)] += 0.001f;
			if (i > 0)
				_A[i + nRows * (i - 1)] += 0.001f;
		}

		A.ReadFrom(_A);
		return cl::smat(A);
	}

	TEST_F(CuSparseTests, Add)
	{
		std::vector<int> _indices = { 0, 5, 10, 50, 75 };
		cl::ivec indices(static_cast<unsigned>(_indices.size()));
		indices.ReadFrom(_indices);

		cl::svec v1(100, indices, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		cl::vec v2 = cl::vec::RandomUniform(v1.denseSize, 1234);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();

		auto v3 = v1 + v2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _v3 = v3.Get();

		for (size_t i = 0; i < v1.size(); ++i)
			ASSERT_TRUE(std::fabs(_v3[i] - _v1[i] - _v2[i]) <= 1e-7f);
	}

	TEST_F(CuSparseTests, Multiply)
	{
		std::vector<int> _NonZeroCols = { 0, 1, 1, 3, 2, 3, 4, 5 };
		cl::ivec gpuNonZeroCols(static_cast<unsigned>(_NonZeroCols.size()));
		gpuNonZeroCols.ReadFrom(_NonZeroCols);

		std::vector<int> _NonZeroRows = { 0, 2, 4, 7, 8 };
		cl::ivec gpuNonZeroRows(static_cast<unsigned>(_NonZeroRows.size()));
		gpuNonZeroRows.ReadFrom(_NonZeroRows);

		cl::smat m1(4, 6, gpuNonZeroCols, gpuNonZeroRows, 1.2345f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m1 = m1.Get();

		cl::mat m2(6, 8, 9.8765f);
		dm::DeviceManager::CheckDeviceSanity();
		auto _m2 = m2.Get();

		auto m3 = m1 * m2;
		dm::DeviceManager::CheckDeviceSanity();
		auto _m3 = m3.Get();

		for (size_t i = 0; i < m1.nRows(); ++i)
		{
			for (size_t j = 0; j < m2.nCols(); ++j)
			{
				double m1m2 = 0.0;
				for (size_t k = 0; k < m1.nCols(); ++k)
					m1m2 += static_cast<double>(_m1[i + k * m1.nRows()] * _m2[k + j * m2.nRows()]);
				ASSERT_LE(std::fabs(m1m2 - static_cast<double>(_m3[i + j * m1.nRows()])), 5e-5);
			}
		}
	}

	TEST_F(CuSparseTests, DISABLED_SolveQr)
	{
		cl::smat v = GetInvertibleSparseMatrix(128);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v = v.Get();

		cl::mat u = cl::mat::RandomGaussian(128, 4, 1234);
		auto _u = u.Get();
		v.Solve(u, LinearSystemSolverType::Qr);
		dm::DeviceManager::CheckDeviceSanity();
		auto _x = u.Get();

		auto uSanity = v.Multiply(u);
		auto _uSanity = uSanity.Get();

		for (size_t i = 0; i < v.nRows(); ++i)
		{
			for (size_t j = 0; j < v.nRows(); ++j)
			{
				float expected = _u[i + v.nRows() * j];
				ASSERT_TRUE(std::fabs(_uSanity[i + v.nRows() * j] - expected) <= 5e-5f) << std::fabs(_uSanity[i + v.nRows() * j] - expected);
			}
		}
	}
}	 // namespace clt
