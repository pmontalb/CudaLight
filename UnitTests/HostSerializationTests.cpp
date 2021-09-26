#include <gtest/gtest.h>

#include <ColumnWiseMatrix.h>
#include <Vector.h>
#include <fstream>
#include <sstream>
#include <cstdio>

namespace clt
{
	class HostSerializationTests: public ::testing::Test
	{
	};

	/*
	 *	If serialization is denoted as f(vec), and deserialization g(vec)
	 *	 ==> f(g(vec)) = vec
	 */
	TEST_F(HostSerializationTests, VectorSerializationInversion)
	{
		std::stringstream s;
		cl::test::vec v(18u, 0.12345f);
		s << v;

		cl::test::vec u = cl::test::vec::VectorFromInputStream(s);

		ASSERT_TRUE(u == v);
	}

	TEST_F(HostSerializationTests, VectorSerializationToBinaryFileInversion)
	{
		cl::test::vec v(18u, 0.12345f);
		v.ToBinaryFile("v1.npy");

		cl::test::vec u = cl::test::vec::VectorFromBinaryFile("v1.npy");

		ASSERT_TRUE(u == v);
	}

	TEST_F(HostSerializationTests, VectorSerializationToBinaryFileInversionCompressed)
	{
		cl::test::vec v(18u, 0.12345f);
		v.ToBinaryFile("v1.npz", true);

		cl::test::vec u = cl::test::vec::VectorFromBinaryFile("v1.npz", true);

		ASSERT_TRUE(u == v);
	}

	/*
	 *	If serialization is denoted as f(vec), and deserialization g(vec)
	 *	 ==> f(g(vec)) = vec
	 */
	TEST_F(HostSerializationTests, MatrixSerializationInversion)
	{
		std::stringstream s;
		cl::test::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		s << m1;

		cl::test::mat m2 = cl::test::mat::MatrixFromInputStream(s);

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
		}
	}

	TEST_F(HostSerializationTests, MatrixSerializationToBinaryFileInversion)
	{
		cl::test::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		m1.ToBinaryFile("m1.npy");

		cl::test::mat m2 = cl::test::mat::MatrixFromBinaryFile("m1.npy");
		m1.Print("m1=");
		m2.Print("m2=");

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
		}
	}

	TEST_F(HostSerializationTests, MatrixSerializationToBinaryFileInversionCompressed)
	{
		cl::test::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		m1.ToBinaryFile("m1.npz", true);

		cl::test::mat m2 = cl::test::mat::MatrixFromBinaryFile("m1.npz", false, true);
		m1.Print("m1=");
		m2.Print("m2=");

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
		}
	}

	/*
	 *	Open file serialized with numpy.savetxt
	 */
	TEST_F(HostSerializationTests, VectorSerializationReadFromNumpy)
	{
		char temp[4096];
		std::string cwd = getcwd(temp, sizeof(temp));
		std::ifstream f("..\\..\\UnitTests\\vec.npy");
		if (!f.is_open())
			f = std::ifstream("vec.npy");
		if (!f.is_open())
			f = std::ifstream(cwd + "/../../UnitTests/vec.npy");
		ASSERT_TRUE(f.is_open());
		ASSERT_FALSE(f.fail());

		try
		{
			cl::test::vec v = cl::test::vec::VectorFromInputStream(f);
			ASSERT_TRUE(v.size() == 128);
			v.Print();
		}
		catch (...)
		{
			ASSERT_FALSE(true);
		}
	}

	/*
	 *	Open file serialized with numpy.savetxt
	 */
	TEST_F(HostSerializationTests, MatrixSerializationReadFromNumpy)
	{
		char temp[4096];
		std::string cwd = getcwd(temp, sizeof(temp));
		std::ifstream f("..\\..\\UnitTests\\mat.npy");
		if (!f.is_open())
			f = std::ifstream("mat.npy");
		if (!f.is_open())
			f = std::ifstream(cwd + "/../../UnitTests/mat.npy");
		ASSERT_TRUE(f.is_open());
		ASSERT_FALSE(f.fail());

		try
		{
			cl::test::mat v = cl::test::mat::MatrixFromInputStream(f);
			ASSERT_TRUE(v.nRows() == 128);
			ASSERT_EQ(v.nCols(), 64);
		}
		catch (...)
		{
			ASSERT_FALSE(true);
		}
	}

	TEST_F(HostSerializationTests, SerializationVectorIntoAFile)
	{
		std::ofstream f("v.cl");
		cl::test::vec v(18u);
		v.LinSpace(0.0f, 1.0f);
		f << v;
		f.close();

		std::remove("v.cl");
	}

	TEST_F(HostSerializationTests, SerializationMatrixIntoAFile)
	{
		std::ofstream f("m.cl");
		cl::test::mat m(18u, 12u);
		m.LinSpace(0.0f, 1.0f);
		f << m;
		f.close();

		std::remove("m.cl");
	}
}	 // namespace clt
