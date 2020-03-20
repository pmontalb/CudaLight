#include <gtest/gtest.h>

#include <HostVector.h>
//#include <HostColumnWiseMatrix.h>
#include <sstream>
#include <fstream>
#include <stdio.h>

namespace clt
{
	class HostSerializationTests : public ::testing::Test
	{
	};

	/*
	*	If serialization is denoted as f(vec), and deserialization g(vec)
	*	 ==> f(g(vec)) = vec
	*/
	TEST_F(HostSerializationTests, VectorSerializationInversion)
	{
		std::stringstream s;
		cl::host::DebugSingleVector v(18u, 0.12345f);
		s << v;

		cl::host::DebugSingleVector u = cl::host::DebugSingleVector::VectorFromInputStream(s);

		ASSERT_TRUE(u == v);
	}

	TEST_F(HostSerializationTests, VectorSerializationToBinaryFileInversion)
	{
		cl::host::DebugSingleVector v(18u, 0.12345f);
		v.ToBinaryFile("v1.npy");

		cl::host::DebugSingleVector u = cl::host::DebugSingleVector::VectorFromBinaryFile("v1.npy");

		ASSERT_TRUE(u == v);
	}
	
	TEST_F(HostSerializationTests, VectorSerializationToBinaryFileInversionCompressed)
	{
		cl::host::DebugSingleVector v(18u, 0.12345f);
		v.ToBinaryFile("v1.npz", true);
		
		cl::host::DebugSingleVector u = cl::host::DebugSingleVector::VectorFromBinaryFile("v1.npz", true);
		
		ASSERT_TRUE(u == v);
	}

//	/*
//	*	If serialization is denoted as f(vec), and deserialization g(vec)
//	*	 ==> f(g(vec)) = vec
//	*/
//	TEST_F(HostSerializationTests, MatrixSerializationInversion)
//	{
//		std::stringstream s;
//		cl::host::DebugSingleColumnWiseMatrix m1(18u, 12u);
//		m1.LinSpace(0.0f, 1.0f);
//		s << m1;
//
//		cl::host::DebugSingleColumnWiseMatrix m2 = cl::host::DebugSingleColumnWiseMatrix::MatrixFromInputStream(s);
//
//		auto _m1 = m1.Get();
//		auto _m2 = m2.Get();
//		for (size_t i = 0; i < m1.nRows(); i++)
//		{
//			for (size_t j = 0; j < m1.nCols(); j++)
//				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
//		}
//	}
//
//	TEST_F(HostSerializationTests, MatrixSerializationToBinaryFileInversion)
//	{
//		cl::host::DebugSingleColumnWiseMatrix m1(18u, 12u);
//		m1.LinSpace(0.0f, 1.0f);
//		m1.ToBinaryFile("m1.npy");
//
//		cl::host::DebugSingleColumnWiseMatrix m2 = cl::host::DebugSingleColumnWiseMatrix::MatrixFromBinaryFile("m1.npy");
//		m1.Print("m1=");
//		m2.Print("m2=");
//
//		auto _m1 = m1.Get();
//		auto _m2 = m2.Get();
//		for (size_t i = 0; i < m1.nRows(); i++)
//		{
//			for (size_t j = 0; j < m1.nCols(); j++)
//				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
//		}
//	}
//	
//	TEST_F(HostSerializationTests, MatrixSerializationToBinaryFileInversionCompressed)
//	{
//		cl::host::DebugSingleColumnWiseMatrix m1(18u, 12u);
//		m1.LinSpace(0.0f, 1.0f);
//		m1.ToBinaryFile("m1.npz", true);
//		
//		cl::host::DebugSingleColumnWiseMatrix m2 = cl::host::DebugSingleColumnWiseMatrix::MatrixFromBinaryFile("m1.npz", true);
//		m1.Print("m1=");
//		m2.Print("m2=");
//		
//		auto _m1 = m1.Get();
//		auto _m2 = m2.Get();
//		for (size_t i = 0; i < m1.nRows(); i++)
//		{
//			for (size_t j = 0; j < m1.nCols(); j++)
//				ASSERT_TRUE(std::fabs(_m1[i + m1.nRows() * j] - _m2[i + m1.nRows() * j]) < 1e-6f);
//		}
//	}

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
            f = std::ifstream(cwd + "/../UnitTests/vec.npy");
		ASSERT_TRUE(f.is_open());
		ASSERT_FALSE(f.fail());

		try
		{
			cl::host::DebugSingleVector v = cl::host::DebugSingleVector::VectorFromInputStream(f);
			ASSERT_TRUE(v.size() == 128);
			v.Print();
		}
		catch (...)
		{
			ASSERT_FALSE(true);
		}		
	}

//	/*
//	*	Open file serialized with numpy.savetxt
//	*/
//	TEST_F(HostSerializationTests, MatrixSerializationReadFromNumpy)
//	{
//		char temp[4096];
//		std::string cwd = getcwd(temp, sizeof(temp));
//		std::ifstream f("..\\..\\UnitTests\\mat.npy");
//		if (!f.is_open())
//			f = std::ifstream("mat.npy");
//        if (!f.is_open())
//            f = std::ifstream(cwd + "/../UnitTests/mat.npy");
//		ASSERT_TRUE(f.is_open());
//		ASSERT_FALSE(f.fail());
//
//		try
//		{
//			cl::host::DebugSingleColumnWiseMatrix v = cl::host::DebugSingleColumnWiseMatrix::MatrixFromInputStream(f);
//			ASSERT_TRUE(v.nRows() == 128);
//			ASSERT_EQ(v.nCols(), 64);
//		}
//		catch (...)
//		{
//			ASSERT_FALSE(true);
//		}
//	}

	TEST_F(HostSerializationTests, SerializationVectorIntoAFile)
	{
		std::ofstream f("v.cl");
		cl::host::DebugSingleVector v(18u);
		v.LinSpace(0.0f, 1.0f);
		f << v;
		f.close();

		std::remove("v.cl");
	}

//	TEST_F(HostSerializationTests, SerializationMatrixIntoAFile)
//	{
//		std::ofstream f("m.cl");
//		cl::host::DebugSingleColumnWiseMatrix m(18u, 12u);
//		m.LinSpace(0.0f, 1.0f);
//		f << m;
//		f.close();
//
//		std::remove("m.cl");
//	}
}
