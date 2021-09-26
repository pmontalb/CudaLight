#pragma once

namespace dm
{
	class DeviceManager
	{
	public:
		static unsigned GetDeviceCount();

		static void SetDevice(const unsigned i);
		static void SetBestDevice();
		static void CheckDeviceSanity();

		DeviceManager(const DeviceManager&) noexcept = delete;
		DeviceManager(DeviceManager&&) noexcept = delete;
		DeviceManager& operator=(const DeviceManager&) noexcept = delete;
		DeviceManager& operator=(DeviceManager&&) noexcept = delete;
	private:
		DeviceManager();
		~DeviceManager() noexcept = default;
	};
}	 // namespace dm
