#include <gtest/gtest.h>

#include <Vector.h>

namespace clt
{
	class CubTests : public ::testing::Test
	{
	};

	TEST_F(CubTests, Sum)
	{
		const size_t size = 1024;
		auto v1 = cl::LinSpace<MemorySpace::Device, MathDomain::Double>(-1.0, 1.0, size);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v1 = v1.Get();

		double sum = v1.Sum();
		dm::DeviceManager::CheckDeviceSanity();
		double sumSanity = 0.0;
		for (size_t i = 0; i < v1.size(); ++i)
			sumSanity += _v1[i];

		ASSERT_NEAR(sumSanity, sum, 1e-12);

		cl::vec v2 = cl::LinSpace(-1.0, 1.0, size + 1);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v2 = v2.Get();

		sum = v2.Sum();
		sumSanity = 0.0;
		for (size_t i = 0; i < v2.size(); ++i)
			sumSanity += _v2[i];

		ASSERT_NEAR(sumSanity, sum, 1e-7);

		cl::ivec v3(size, 1);
		dm::DeviceManager::CheckDeviceSanity();
		auto _v3 = v3.Get();

		sum = v3.Sum();
		sumSanity = 0.0;
		for (size_t i = 0; i < v3.size(); ++i)
			sumSanity += _v3[i];

		ASSERT_NEAR(sumSanity, sum, 0);
	}
}
