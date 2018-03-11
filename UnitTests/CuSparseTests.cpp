#include "stdafx.h"
#include "CppUnitTest.h"

#include <SparseVector.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(CuSparseTests)
	{
	public:
		TEST_METHOD(Add)
		{
			std::vector<int> _indices = { 0, 5, 10, 50, 75 };
			cl::ivec indices(static_cast<unsigned>(_indices.size()));
			indices.ReadFrom(_indices);

			cl::svec v1(100, indices, 1.2345f);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v1 = v1.Get();

			cl::vec v2 = cl::RandomUniform(v1.denseSize);
			dm::DeviceManager::CheckDeviceSanity();
			auto _v2 = v2.Get();

			auto v3 = v1 + v2;
			dm::DeviceManager::CheckDeviceSanity();
			auto _v3 = v3.Get();

			for (size_t i = 0; i < v1.size(); ++i)
				Assert::IsTrue(fabs(_v3[i] - _v1[i] - _v2[i]) <= 1e-7);
		}
	};
}