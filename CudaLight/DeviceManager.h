#pragma once

#include <DeviceManagerHelper.h>

namespace dm
{
	class DeviceManager 
	{
	public:
		static unsigned GetDeviceCount();

		static void SetDevice(const unsigned i);
		static void SetBestDevice();
		static void CheckDeviceSanity();

	private:
		DeviceManager();
		~DeviceManager() noexcept = default;

		DeviceManager(const DeviceManager&) noexcept = delete;
		DeviceManager(DeviceManager&&) noexcept = delete;
		DeviceManager& operator=(const DeviceManager&) noexcept = delete;
		DeviceManager& operator=(DeviceManager&&) noexcept = delete;
	};
}

