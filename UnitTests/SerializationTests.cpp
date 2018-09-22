#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <sstream>
#include <fstream>
#include <stdio.h>

namespace clt
{
	class SerializationTests : public ::testing::Test
	{
	};

	/*
	*	If serialization is denoted as f(vec), and deserialization g(vec)
	*	 ==> f(g(vec)) = vec
	*/
	TEST_F(SerializationTests, VectorSerializationInversion)
	{
		std::stringstream s;
		cl::vec v(18u, 0.12345f);
		s << v;

		cl::vec u = cl::VectorFromInputStream(s);

		ASSERT_TRUE(u == v);
	}

	TEST_F(SerializationTests, VectorSerializationToBinaryFileInversion)
	{
		cl::vec v(18u, 0.12345f);
		v.ToBinaryFile("v1.npy");

		cl::vec u = cl::VectorFromBinaryFile("v1.npy");

		ASSERT_TRUE(u == v);
	}

	/*
	*	If serialization is denoted as f(vec), and deserialization g(vec)
	*	 ==> f(g(vec)) = vec
	*/
	TEST_F(SerializationTests, MatrixSerializationInversion)
	{
		std::stringstream s;
		cl::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		s << m1;

		cl::mat m2 = cl::MatrixFromInputStream(s);

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6);
		}
	}

	TEST_F(SerializationTests, MatrixSerializationToBinaryFileInversion)
	{
		cl::mat m1(18u, 12u);
		m1.LinSpace(0.0f, 1.0f);
		m1.ToBinaryFile("m1.npy");

		cl::mat m2 = cl::MatrixFromBinaryFile("m1.npy");

		auto _m1 = m1.Get();
		auto _m2 = m2.Get();
		for (size_t i = 0; i < m1.nRows(); i++)
		{
			for (size_t j = 0; j < m1.nCols(); j++)
				ASSERT_TRUE(fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6);
		}
	}

	/*
	*	Open file serialized with numpy.savetxt
	*/
	TEST_F(SerializationTests, VectorSerializationReadFromNumpy)
	{
		char temp[4096];
		std::string cwd = getcwd(temp, sizeof(temp));
		std::ifstream f("..\\..\\UnitTests\\vec.npy");
		if (!f.is_open())
			f = std::ifstream("vec.npy");
		ASSERT_TRUE(f.is_open());
		ASSERT_FALSE(f.fail());

		try
		{
			cl::vec v = cl::VectorFromInputStream(f);
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
	TEST_F(SerializationTests, MatrixSerializationReadFromNumpy)
	{
		char temp[4096];
		std::string cwd = getcwd(temp, sizeof(temp));
		std::ifstream f("..\\..\\UnitTests\\mat.npy");
		if (!f.is_open())
			f = std::ifstream("mat.npy");
		ASSERT_TRUE(f.is_open());
		ASSERT_FALSE(f.fail());

		try
		{
			cl::mat v = cl::MatrixFromInputStream(f);
			ASSERT_TRUE(v.nRows() == 128);
			ASSERT_TRUE(v.nCols() == 64);
		}
		catch (...)
		{
			ASSERT_FALSE(true);
		}
	}

	TEST_F(SerializationTests, SerializationVectorIntoAFile)
	{
		std::ofstream f("v.cl");
		cl::vec v(18u);
		v.LinSpace(0.0f, 1.0f);
		f << v;
		f.close();

		std::remove("v.cl");
	}

	TEST_F(SerializationTests, SerializationMatrixIntoAFile)
	{
		std::ofstream f("m.cl");
		cl::mat m(18u, 12u);
		m.LinSpace(0.0f, 1.0f);
		f << m;
		f.close();

		std::remove("m.cl");
	}
}