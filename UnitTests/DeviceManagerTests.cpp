#include "stdafx.h"
#include "CppUnitTest.h"
#include <DeviceManager.h>
#include <DeviceManager.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{		
	TEST_CLASS(DeviceManagerTests)
	{
	public:
		
		TEST_METHOD(DeviceInitialization)
		{
			for (int i = 0; i < dm::DeviceManager::GetDeviceCount(); ++i)
			{
				dm::DeviceManager::SetDevice(i);
				dm::DeviceManager::CheckDeviceSanity();
			}

			dm::DeviceManager::SetBestDevice();
			dm::DeviceManager::CheckDeviceSanity();
		}
	};
}