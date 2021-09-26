#include <DeviceManager.h>
#include <DeviceManagerHelper.h>

namespace dm
{
	DeviceManager::DeviceManager() { SetBestDevice(); }

	void DeviceManager::SetDevice(const unsigned i)
	{
		detail::SetDevice(static_cast<int>(i));
		CheckDeviceSanity();
	}

	void DeviceManager::SetBestDevice()
	{
		int bestDevice = -1;
		detail::GetBestDevice(bestDevice);
		SetDevice(static_cast<unsigned>(bestDevice));
	}

	void DeviceManager::CheckDeviceSanity() { detail::GetDeviceStatus(); }

	unsigned DeviceManager::GetDeviceCount()
	{
		int ret = -1;
		detail::GetDeviceCount(ret);

		return static_cast<unsigned>(ret);
	}
}	 // namespace dm
