#include <DeviceManager.h>
#include <DeviceManagerHelper.h>

namespace dm
{
	DeviceManager::DeviceManager()
	{
		SetBestDevice();
	}

	void DeviceManager::SetDevice(const unsigned i)
	{
		detail::SetDevice(i);
		CheckDeviceSanity();
	}

	void DeviceManager::SetBestDevice()
	{
		int bestDevice = -1;
		detail::GetBestDevice(bestDevice);
		SetDevice(bestDevice);
	}

	void DeviceManager::CheckDeviceSanity()
	{	
		detail::GetDeviceStatus();
	}

	unsigned DeviceManager::GetDeviceCount()
	{
		int ret;
		detail::GetDeviceCount(ret);

		return ret;
	}
}
