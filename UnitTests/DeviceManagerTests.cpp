#include <DeviceManager.h>
#include <gtest/gtest.h>

namespace clt
{
	class DeviceManagerTests: public ::testing::Test
	{
	};

	TEST_F(DeviceManagerTests, DeviceInitialization)
	{
		for (unsigned i = 0; i < dm::DeviceManager::GetDeviceCount(); ++i)
		{
			dm::DeviceManager::SetDevice(i);
			dm::DeviceManager::CheckDeviceSanity();
		}

		dm::DeviceManager::SetBestDevice();
		dm::DeviceManager::CheckDeviceSanity();
	}
}	 // namespace clt
