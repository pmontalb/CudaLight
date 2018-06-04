#pragma once

#include <Exception.h>

class CudaGenericKernelException : public Exception
{
public:
	CudaGenericKernelException(const std::string& kernelName, const int errorCode = -1)
		: Exception(kernelName + " returned " + std::to_string(errorCode))
	{
	}
};

#pragma region Cuda Exception mapping

class CudaErrorMissingConfigurationException: public Exception 
{
public:
	CudaErrorMissingConfigurationException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The API call failed because it was unable to allocate enough memory to
* perform the requested operation.
*/
class CudaErrorMemoryAllocationException: public Exception 
{
public:
	CudaErrorMemoryAllocationException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The API call failed because the CUDA driver and runtime could not be
* initialized.
*/
class CudaErrorInitializationErrorException: public Exception 
{
public:
	CudaErrorInitializationErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* An exception occurred on the device while executing a kernel. Common
* causes include dereferencing an invalid device pointer and accessing
* out of bounds shared memory. The device cannot be used until
* ::cudaThreadExit() is called. All existing device memory allocations
* are invalid and must be reconstructed if the program is to continue
* using CUDA.
*/
class CudaErrorLaunchFailureException: public Exception 
{
public:
	CudaErrorLaunchFailureException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that a previous kernel launch failed. This was previously
* used for device emulation of kernel launches.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorPriorLaunchFailureException: public Exception 
{
public:
	CudaErrorPriorLaunchFailureException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the device kernel took too long to execute. This can
* only occur if timeouts are enabled - see the device property
* \ref ::cudaDeviceProp::kernelExecTimeoutEnabled kernelExecTimeoutEnabled
* for more information.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorLaunchTimeoutException: public Exception 
{
public:
	CudaErrorLaunchTimeoutException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a launch did not occur because it did not have
* appropriate resources. Although this error is similar to
* ::cudaErrorInvalidConfiguration, this error usually indicates that the
* user has attempted to pass too many arguments to the device kernel, or the
* kernel launch specifies too many threads for the kernel's register count.
*/
class CudaErrorLaunchOutOfResourcesException: public Exception 
{
public:
	CudaErrorLaunchOutOfResourcesException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The requested device function does not exist or is not compiled for the
* proper device architecture.
*/
class CudaErrorInvalidDeviceFunctionException: public Exception 
{
public:
	CudaErrorInvalidDeviceFunctionException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a kernel launch is requesting resources that can
* never be satisfied by the current device. Requesting more shared memory
* per block than the device supports will trigger this error, as will
* requesting too many threads or blocks. See ::cudaDeviceProp for more
* device limitations.
*/
class CudaErrorInvalidConfigurationException: public Exception 
{
public:
	CudaErrorInvalidConfigurationException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the device ordinal supplied by the user does not
* correspond to a valid CUDA device.
*/
class CudaErrorInvalidDeviceException: public Exception 
{
public:
	CudaErrorInvalidDeviceException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that one or more of the parameters passed to the API call
* is not within an acceptable range of values.
*/
class CudaErrorInvalidValueException: public Exception 
{
public:
	CudaErrorInvalidValueException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that one or more of the pitch-related parameters passed
* to the API call is not within the acceptable range for pitch.
*/
class CudaErrorInvalidPitchValueException: public Exception 
{
public:
	CudaErrorInvalidPitchValueException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the symbol name/identifier passed to the API call
* is not a valid name or identifier.
*/
class CudaErrorInvalidSymbolException: public Exception 
{
public:
	CudaErrorInvalidSymbolException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the buffer object could not be mapped.
*/
class CudaErrorMapBufferObjectFailedException: public Exception 
{
public:
	CudaErrorMapBufferObjectFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the buffer object could not be unmapped.
*/
class CudaErrorUnmapBufferObjectFailedException: public Exception 
{
public:
	CudaErrorUnmapBufferObjectFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that at least one host pointer passed to the API call is
* not a valid host pointer.
*/
class CudaErrorInvalidHostPointerException: public Exception 
{
public:
	CudaErrorInvalidHostPointerException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that at least one device pointer passed to the API call is
* not a valid device pointer.
*/
class CudaErrorInvalidDevicePointerException: public Exception 
{
public:
	CudaErrorInvalidDevicePointerException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the texture passed to the API call is not a valid
* texture.
*/
class CudaErrorInvalidTextureException: public Exception 
{
public:
	CudaErrorInvalidTextureException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the texture binding is not valid. This occurs if you
* call ::cudaGetTextureAlignmentOffset() with an unbound texture.
*/
class CudaErrorInvalidTextureBindingException: public Exception 
{
public:
	CudaErrorInvalidTextureBindingException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the channel descriptor passed to the API call is not
* valid. This occurs if the format is not one of the formats specified by
* ::cudaChannelFormatKind, or if one of the dimensions is invalid.
*/
class CudaErrorInvalidChannelDescriptorException: public Exception 
{
public:
	CudaErrorInvalidChannelDescriptorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the direction of the memcpy passed to the API call is
* not one of the types specified by ::cudaMemcpyKind.
*/
class CudaErrorInvalidMemcpyDirectionException: public Exception 
{
public:
	CudaErrorInvalidMemcpyDirectionException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that the user has taken the address of a constant variable,
* which was forbidden up until the CUDA 3.1 release.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Variables in constant
* memory may now have their address taken by the runtime via
* ::cudaGetSymbolAddress().
*/
class CudaErrorAddressOfConstantException: public Exception 
{
public:
	CudaErrorAddressOfConstantException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that a texture fetch was not able to be performed.
* This was previously used for device emulation of texture operations.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorTextureFetchFailedException: public Exception 
{
public:
	CudaErrorTextureFetchFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that a texture was not bound for access.
* This was previously used for device emulation of texture operations.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorTextureNotBoundException: public Exception 
{
public:
	CudaErrorTextureNotBoundException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that a synchronization operation had failed.
* This was previously used for some device emulation functions.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorSynchronizationErrorException: public Exception 
{
public:
	CudaErrorSynchronizationErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a non-float texture was being accessed with linear
* filtering. This is not supported by CUDA.
*/
class CudaErrorInvalidFilterSettingException: public Exception 
{
public:
	CudaErrorInvalidFilterSettingException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that an attempt was made to read a non-float texture as a
* normalized float. This is not supported by CUDA.
*/
class CudaErrorInvalidNormSettingException: public Exception 
{
public:
	CudaErrorInvalidNormSettingException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* Mixing of device and device emulation code was not allowed.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorMixedDeviceExecutionException: public Exception 
{
public:
	CudaErrorMixedDeviceExecutionException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a CUDA Runtime API call cannot be executed because
* it is being called during process shut down, at a point in time after
* CUDA driver has been unloaded.
*/
class CudaErrorCudartUnloadingException: public Exception 
{
public:
	CudaErrorCudartUnloadingException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that an unknown internal error has occurred.
*/
class CudaErrorUnknownException: public Exception 
{
public:
	CudaErrorUnknownException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the API call is not yet implemented. Production
* releases of CUDA will never return this error.
* \deprecated
* This error return is deprecated as of CUDA 4.1.
*/
class CudaErrorNotYetImplementedException: public Exception 
{
public:
	CudaErrorNotYetImplementedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicated that an emulated device pointer exceeded the 32-bit address
* range.
* \deprecated
* This error return is deprecated as of CUDA 3.1. Device emulation mode was
* removed with the CUDA 3.1 release.
*/
class CudaErrorMemoryValueTooLargeException: public Exception 
{
public:
	CudaErrorMemoryValueTooLargeException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a resource handle passed to the API call was not
* valid. Resource handles are opaque types like ::cudaStream_t and
* ::cudaEvent_t.
*/
class CudaErrorInvalidResourceHandleException: public Exception 
{
public:
	CudaErrorInvalidResourceHandleException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that asynchronous operations issued previously have not
* completed yet. This result is not actually an error, but must be indicated
* differently than ::cudaSuccess (which indicates completion). Calls that
* may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
*/
class CudaErrorNotReadyException: public Exception 
{
public:
	CudaErrorNotReadyException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the installed NVIDIA CUDA driver is older than the
* CUDA runtime library. This is not a supported configuration. Users should
* install an updated NVIDIA display driver to allow the application to run.
*/
class CudaErrorInsufficientDriverException: public Exception 
{
public:
	CudaErrorInsufficientDriverException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the user has called ::cudaSetValidDevices(),
* ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
* ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
* ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
* calling non-device management operations (allocating memory and
* launching kernels are examples of non-device management operations).
* This error can also be returned if using runtime/driver
* interoperability and there is an existing ::CUcontext active on the
* host thread.
*/
class CudaErrorSetOnActiveProcessException: public Exception 
{
public:
	CudaErrorSetOnActiveProcessException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the surface passed to the API call is not a valid
* surface.
*/
class CudaErrorInvalidSurfaceException: public Exception 
{
public:
	CudaErrorInvalidSurfaceException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that no CUDA-capable devices were detected by the installed
* CUDA driver.
*/
class CudaErrorNoDeviceException: public Exception 
{
public:
	CudaErrorNoDeviceException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that an uncorrectable ECC error was detected during
* execution.
*/
class CudaErrorECCUncorrectableException: public Exception 
{
public:
	CudaErrorECCUncorrectableException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a link to a shared object failed to resolve.
*/
class CudaErrorSharedObjectSymbolNotFoundException: public Exception 
{
public:
	CudaErrorSharedObjectSymbolNotFoundException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that initialization of a shared object failed.
*/
class CudaErrorSharedObjectInitFailedException: public Exception 
{
public:
	CudaErrorSharedObjectInitFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the ::cudaLimit passed to the API call is not
* supported by the active device.
*/
class CudaErrorUnsupportedLimitException: public Exception 
{
public:
	CudaErrorUnsupportedLimitException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that multiple global or constant variables (across separate
* CUDA source files in the application) share the same string name.
*/
class CudaErrorDuplicateVariableNameException: public Exception 
{
public:
	CudaErrorDuplicateVariableNameException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that multiple textures (across separate CUDA source
* files in the application) share the same string name.
*/
class CudaErrorDuplicateTextureNameException: public Exception 
{
public:
	CudaErrorDuplicateTextureNameException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that multiple surfaces (across separate CUDA source
* files in the application) share the same string name.
*/
class CudaErrorDuplicateSurfaceNameException: public Exception 
{
public:
	CudaErrorDuplicateSurfaceNameException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that all CUDA devices are busy or unavailable at the current
* time. Devices are often busy/unavailable due to use of
* ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
* running CUDA kernels have filled up the GPU and are blocking new work
* from starting. They can also be unavailable due to memory constraints
* on a device that already has active CUDA work being performed.
*/
class CudaErrorDevicesUnavailableException: public Exception 
{
public:
	CudaErrorDevicesUnavailableException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the device kernel image is invalid.
*/
class CudaErrorInvalidKernelImageException: public Exception 
{
public:
	CudaErrorInvalidKernelImageException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that there is no kernel image available that is suitable
* for the device. This can occur when a user specifies code generation
* options for a particular CUDA source file that do not include the
* corresponding device configuration.
*/
class CudaErrorNoKernelImageForDeviceException: public Exception 
{
public:
	CudaErrorNoKernelImageForDeviceException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the current context is not compatible with this
* the CUDA Runtime. This can only occur if you are using CUDA
* Runtime/Driver interoperability and have created an existing Driver
* context using the driver API. The Driver context may be incompatible
* either because the Driver context was created using an older version
* of the API, because the Runtime API call expects a primary driver
* context and the Driver context is not primary, or because the Driver
* context has been destroyed. Please see \ref CUDART_DRIVER Interactions
* with the CUDA Driver API for more information.
*/
class CudaErrorIncompatibleDriverContextException: public Exception 
{
public:
	CudaErrorIncompatibleDriverContextException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
* trying to re-enable peer addressing on from a context which has already
* had peer addressing enabled.
*/
class CudaErrorPeerAccessAlreadyEnabledException: public Exception 
{
public:
	CudaErrorPeerAccessAlreadyEnabledException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that ::cudaDeviceDisablePeerAccess() is trying to
* disable peer addressing which has not been enabled yet via
* ::cudaDeviceEnablePeerAccess().
*/
class CudaErrorPeerAccessNotEnabledException: public Exception 
{
public:
	CudaErrorPeerAccessNotEnabledException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that a call tried to access an exclusive-thread device that
* is already in use by a different thread.
*/
class CudaErrorDeviceAlreadyInUseException: public Exception 
{
public:
	CudaErrorDeviceAlreadyInUseException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates profiler is not initialized for this run. This can
* happen when the application is running with external profiling tools
* like visual profiler.
*/
class CudaErrorProfilerDisabledException: public Exception 
{
public:
	CudaErrorProfilerDisabledException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* \deprecated
* This error return is deprecated as of CUDA 5.0. It is no longer an error
* to attempt to enable/disable the profiling via ::cudaProfilerStart or
* ::cudaProfilerStop without initialization.
*/
class CudaErrorProfilerNotInitializedException: public Exception 
{
public:
	CudaErrorProfilerNotInitializedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* \deprecated
* This error return is deprecated as of CUDA 5.0. It is no longer an error
* to call cudaProfilerStart() when profiling is already enabled.
*/
class CudaErrorProfilerAlreadyStartedException: public Exception 
{
public:
	CudaErrorProfilerAlreadyStartedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* \deprecated
* This error return is deprecated as of CUDA 5.0. It is no longer an error
* to call cudaProfilerStop() when profiling is already disabled.
*/
class CudaErrorProfilerAlreadyStoppedException: public Exception 
{
public:
	CudaErrorProfilerAlreadyStoppedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* An assert triggered in device code during kernel execution. The device
* cannot be used again until ::cudaThreadExit() is called. All existing
* allocations are invalid and must be reconstructed if the program is to
* continue using CUDA.
*/
class CudaErrorAssertException: public Exception 
{
public:
	CudaErrorAssertException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that the hardware resources required to enable
* peer access have been exhausted for one or more of the devices
* passed to ::cudaEnablePeerAccess().
*/
class CudaErrorTooManyPeersException: public Exception 
{
public:
	CudaErrorTooManyPeersException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that the memory range passed to ::cudaHostRegister()
* has already been registered.
*/
class CudaErrorHostMemoryAlreadyRegisteredException: public Exception 
{
public:
	CudaErrorHostMemoryAlreadyRegisteredException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that the pointer passed to ::cudaHostUnregister()
* does not correspond to any currently registered memory region.
*/
class CudaErrorHostMemoryNotRegisteredException: public Exception 
{
public:
	CudaErrorHostMemoryNotRegisteredException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that an OS call failed.
*/
class CudaErrorOperatingSystemException: public Exception 
{
public:
	CudaErrorOperatingSystemException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that P2P access is not supported across the given
* devices.
*/
class CudaErrorPeerAccessUnsupportedException: public Exception 
{
public:
	CudaErrorPeerAccessUnsupportedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a device runtime grid launch did not occur
* because the depth of the child grid would exceed the maximum supported
* number of nested grid launches.
*/
class CudaErrorLaunchMaxDepthExceededException: public Exception 
{
public:
	CudaErrorLaunchMaxDepthExceededException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a grid launch did not occur because the kernel
* uses file-scoped textures which are unsupported by the device runtime.
* Kernels launched via the device runtime only support textures created with
* the Texture Object API's.
*/
class CudaErrorLaunchFileScopedTexException: public Exception 
{
public:
	CudaErrorLaunchFileScopedTexException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a grid launch did not occur because the kernel
* uses file-scoped surfaces which are unsupported by the device runtime.
* Kernels launched via the device runtime only support surfaces created with
* the Surface Object API's.
*/
class CudaErrorLaunchFileScopedSurfException: public Exception 
{
public:
	CudaErrorLaunchFileScopedSurfException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a call to ::cudaDeviceSynchronize made from
* the device runtime failed because the call was made at grid depth greater
* than than either the default (2 levels of grids) or user specified device
* limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
* launched grids at a greater depth successfully, the maximum nested
* depth at which ::cudaDeviceSynchronize will be called must be specified
* with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
* api before the host-side launch of a kernel using the device runtime.
* Keep in mind that additional levels of sync depth require the runtime
* to reserve large amounts of device memory that cannot be used for
* user allocations.
*/
class CudaErrorSyncDepthExceededException: public Exception 
{
public:
	CudaErrorSyncDepthExceededException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that a device runtime grid launch failed because
* the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
* For this launch to proceed successfully, ::cudaDeviceSetLimit must be
* called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher
* than the upper bound of outstanding launches that can be issued to the
* device runtime. Keep in mind that raising the limit of pending device
* runtime launches will require the runtime to reserve device memory that
* cannot be used for user allocations.
*/
class CudaErrorLaunchPendingCountExceededException: public Exception 
{
public:
	CudaErrorLaunchPendingCountExceededException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates the attempted operation is not permitted.
*/
class CudaErrorNotPermittedException: public Exception 
{
public:
	CudaErrorNotPermittedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates the attempted operation is not supported
* on the current system or device.
*/
class CudaErrorNotSupportedException: public Exception 
{
public:
	CudaErrorNotSupportedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* Device encountered an error in the call stack during kernel execution,
* possibly due to stack corruption or exceeding the stack size limit.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorHardwareStackErrorException: public Exception 
{
public:
	CudaErrorHardwareStackErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The device encountered an illegal instruction during kernel execution
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorIllegalInstructionException: public Exception 
{
public:
	CudaErrorIllegalInstructionException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The device encountered a load or store instruction
* on a memory address which is not aligned.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorMisalignedAddressException: public Exception 
{
public:
	CudaErrorMisalignedAddressException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* While executing a kernel, the device encountered an instruction
* which can only operate on memory locations in certain address spaces
* (global, shared, or local), but was supplied a memory address not
* belonging to an allowed address space.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorInvalidAddressSpaceException: public Exception 
{
public:
	CudaErrorInvalidAddressSpaceException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The device encountered an invalid program counter.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorInvalidPcException: public Exception 
{
public:
	CudaErrorInvalidPcException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* The device encountered a load or store instruction on an invalid memory address.
* This leaves the process in an inconsistent state and any further CUDA work
* will return the same error. To continue using CUDA, the process must be terminated
* and relaunched.
*/
class CudaErrorIllegalAddressException: public Exception 
{
public:
	CudaErrorIllegalAddressException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* A PTX compilation failed. The runtime may fall back to compiling PTX if
* an application does not contain a suitable binary for the current device.
*/
class CudaErrorInvalidPtxException: public Exception 
{
public:
	CudaErrorInvalidPtxException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates an error with the OpenGL or DirectX context.
*/
class CudaErrorInvalidGraphicsContextException: public Exception 
{
public:
	CudaErrorInvalidGraphicsContextException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that an uncorrectable NVLink error was detected during the
* execution.
*/
class CudaErrorNvlinkUncorrectableException: public Exception 
{
public:
	CudaErrorNvlinkUncorrectableException(const std::string& kernelName)  : Exception(kernelName)
	{
	}
}; 

/**
* This indicates that the PTX JIT compiler library was not found. The JIT Compiler
* library is used for PTX compilation. The runtime may fall back to compiling PTX
* if an application does not contain a suitable binary for the current device.
*/
class CudaErrorJitCompilerNotFoundException: public Exception 
{
public:
	CudaErrorJitCompilerNotFoundException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This error indicates that the number of blocks launched per grid for a kernel that was
* launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
* exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
* or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
* as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
*/
class CudaErrorCooperativeLaunchTooLargeException: public Exception 
{
public:
	CudaErrorCooperativeLaunchTooLargeException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* This indicates an internal startup failure in the CUDA runtime.
*/
class CudaErrorStartupFailureException: public Exception 
{
public:
	CudaErrorStartupFailureException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

/**
* Any unhandled CUDA driver error is added to this value and returned via
* the runtime. Production releases of CUDA should not return such errors.
* \deprecated
* This error return is deprecated as of CUDA 4.1.
*/
class CudaErrorApiFailureBaseException: public Exception 
{
public:
	CudaErrorApiFailureBaseException(const std::string& kernelName) : Exception(kernelName)
	{
	}
}; 

#pragma endregion

#pragma region CuBlas Exception mapping

class CuBlasNotInitialisedException : public Exception
{
public:
	CuBlasNotInitialisedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasAllocFailedException : public Exception
{
public:
	CuBlasAllocFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasInvalidValueException : public Exception
{
public:
	CuBlasInvalidValueException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasArchMismatchException : public Exception
{
public:
	CuBlasArchMismatchException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasMappingErrorException : public Exception
{
public:
	CuBlasMappingErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasExecutionFailedException : public Exception
{
public:
	CuBlasExecutionFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasInternalErrorException : public Exception
{
public:
	CuBlasInternalErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasNotSupportedException : public Exception
{
public:
	CuBlasNotSupportedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuBlasLicenseErrorException : public Exception
{
public:
	CuBlasLicenseErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

#pragma endregion

#pragma region CuSparse Exception mapping

class CuSparseNotInitialisedException : public Exception
{
public:
	CuSparseNotInitialisedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseAllocFailedException : public Exception
{
public:
	CuSparseAllocFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseInvalidValueException : public Exception
{
public:
	CuSparseInvalidValueException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseArchMismatchException : public Exception
{
public:
	CuSparseArchMismatchException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseMappingErrorException : public Exception
{
public:
	CuSparseMappingErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseExecutionFailedException : public Exception
{
public:
	CuSparseExecutionFailedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseInternalErrorException : public Exception
{
public:
	CuSparseInternalErrorException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseMatrixTypeNotSupportedException : public Exception
{
public:
	CuSparseMatrixTypeNotSupportedException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

class CuSparseZeroPivotException : public Exception
{
public:
	CuSparseZeroPivotException(const std::string& kernelName) : Exception(kernelName)
	{
	}
};

#pragma endregion


// CudaLightInternalExceptions
struct CudaGenericKernelExceptionFactory
{
	static void ThrowException(const std::string& kernelName, const int errorCode)
	{
		switch (errorCode)
		{
			case -4:
				throw InternalErrorException(kernelName);
			case -3:
				throw ExpectedEvenSizeException(kernelName);
			case -2:
				throw NotSupportedException(kernelName);
			case -1:
			default:
				throw NotImplementedException(kernelName);
		}
	}
};

struct CudaKernelExceptionFactory
{
	static void ThrowException(const std::string& kernelName, const int errorCode)
	{
		switch (errorCode)
		{
			// CudaLightInternalExceptions
			case -3:
			case -2:
			case -1:
				CudaGenericKernelExceptionFactory::ThrowException(kernelName, errorCode);
			case 1:
				throw CudaErrorMissingConfigurationException(kernelName);
			case 2:
				throw CudaErrorMemoryAllocationException(kernelName);
			case 3:
				throw CudaErrorInitializationErrorException(kernelName);
			case 4:
				throw CudaErrorLaunchFailureException(kernelName);
			case 5:
				throw CudaErrorPriorLaunchFailureException(kernelName);
			case 6:
				throw CudaErrorLaunchTimeoutException(kernelName);
			case 7:
				throw CudaErrorLaunchOutOfResourcesException(kernelName);
			case 8:
				throw CudaErrorInvalidDeviceFunctionException(kernelName);
			case 9:
				throw CudaErrorInvalidConfigurationException(kernelName);
			case 10:
				throw CudaErrorInvalidDeviceException(kernelName);
			case 11:
				throw CudaErrorInvalidValueException(kernelName);
			case 12:
				throw CudaErrorInvalidPitchValueException(kernelName);
			case 13:
				throw CudaErrorInvalidSymbolException(kernelName);
			case 14:
				throw CudaErrorMapBufferObjectFailedException(kernelName);
			case 15:
				throw CudaErrorUnmapBufferObjectFailedException(kernelName);
			case 16:
				throw CudaErrorInvalidHostPointerException(kernelName);
			case 17:
				throw CudaErrorInvalidDevicePointerException(kernelName);
			case 18:
				throw CudaErrorInvalidTextureException(kernelName);
			case 19:
				throw CudaErrorInvalidTextureBindingException(kernelName);
			case 20:
				throw CudaErrorInvalidChannelDescriptorException(kernelName);
			case 21:
				throw CudaErrorInvalidMemcpyDirectionException(kernelName);
			case 22:
				throw CudaErrorAddressOfConstantException(kernelName);
			case 23:
				throw CudaErrorTextureFetchFailedException(kernelName);
			case 24:
				throw CudaErrorTextureNotBoundException(kernelName);
			case 25:
				throw CudaErrorSynchronizationErrorException(kernelName);
			case 26:
				throw CudaErrorInvalidFilterSettingException(kernelName);
			case 27:
				throw CudaErrorInvalidNormSettingException(kernelName);
			case 28:
				throw CudaErrorMixedDeviceExecutionException(kernelName);
			case 29:
				throw CudaErrorCudartUnloadingException(kernelName);
			case 30:
				throw CudaErrorUnknownException(kernelName);
			case 31:
				throw CudaErrorNotYetImplementedException(kernelName);
			case 32:
				throw CudaErrorMemoryValueTooLargeException(kernelName);
			case 33:
				throw CudaErrorInvalidResourceHandleException(kernelName);
			case 34:
				throw CudaErrorNotReadyException(kernelName);
			case 35:
				throw CudaErrorInsufficientDriverException(kernelName);
			case 36:
				throw CudaErrorSetOnActiveProcessException(kernelName);
			case 37:
				throw CudaErrorInvalidSurfaceException(kernelName);
			case 38:
				throw CudaErrorNoDeviceException(kernelName);
			case 39:
				throw CudaErrorECCUncorrectableException(kernelName);
			case 40:
				throw CudaErrorSharedObjectSymbolNotFoundException(kernelName);
			case 41:
				throw CudaErrorSharedObjectInitFailedException(kernelName);
			case 42:
				throw CudaErrorUnsupportedLimitException(kernelName);
			case 43:
				throw CudaErrorDuplicateVariableNameException(kernelName);
			case 44:
				throw CudaErrorDuplicateTextureNameException(kernelName);
			case 45:
				throw CudaErrorDuplicateSurfaceNameException(kernelName);
			case 46:
				throw CudaErrorDevicesUnavailableException(kernelName);
			case 47:
				throw CudaErrorInvalidKernelImageException(kernelName);
			case 48:
				throw CudaErrorNoKernelImageForDeviceException(kernelName);
			case 49:
				throw CudaErrorIncompatibleDriverContextException(kernelName);
			case 50:
				throw CudaErrorPeerAccessAlreadyEnabledException(kernelName);
			case 51:
				throw CudaErrorPeerAccessNotEnabledException(kernelName);
			case 54:
				throw CudaErrorDeviceAlreadyInUseException(kernelName);
			case 55:
				throw CudaErrorProfilerDisabledException(kernelName);
			case 56:
				throw CudaErrorProfilerNotInitializedException(kernelName);
			case 57:
				throw CudaErrorProfilerAlreadyStartedException(kernelName);
			case 58:
				throw CudaErrorProfilerAlreadyStoppedException(kernelName);
			case 59:
				throw CudaErrorAssertException(kernelName);
			case 60:
				throw CudaErrorTooManyPeersException(kernelName);
			case 61:
				throw CudaErrorHostMemoryAlreadyRegisteredException(kernelName);
			case 62:
				throw CudaErrorHostMemoryNotRegisteredException(kernelName);
			case 63:
				throw CudaErrorOperatingSystemException(kernelName);
			case 64:
				throw CudaErrorPeerAccessUnsupportedException(kernelName);
			case 65:
				throw CudaErrorLaunchMaxDepthExceededException(kernelName);
			case 66:
				throw CudaErrorLaunchFileScopedTexException(kernelName);
			case 67:
				throw CudaErrorLaunchFileScopedSurfException(kernelName);
			case 68:
				throw CudaErrorSyncDepthExceededException(kernelName);
			case 69:
				throw CudaErrorLaunchPendingCountExceededException(kernelName);
			case 70:
				throw CudaErrorNotPermittedException(kernelName);
			case 71:
				throw CudaErrorNotSupportedException(kernelName);
			case 72:
				throw CudaErrorHardwareStackErrorException(kernelName);
			case 73:
				throw CudaErrorIllegalInstructionException(kernelName);
			case 74:
				throw CudaErrorMisalignedAddressException(kernelName);
			case 75:
				throw CudaErrorInvalidAddressSpaceException(kernelName);
			case 76:
				throw CudaErrorInvalidPcException(kernelName);
			case 77:
				throw CudaErrorIllegalAddressException(kernelName);
			case 78:
				throw CudaErrorInvalidPtxException(kernelName);
			case 79:
				throw CudaErrorInvalidGraphicsContextException(kernelName);
			case 80:
				throw CudaErrorNvlinkUncorrectableException(kernelName);
			case 81:
				throw CudaErrorJitCompilerNotFoundException(kernelName);
			case 82:
				throw CudaErrorCooperativeLaunchTooLargeException(kernelName);
			case 0x7f:
				throw CudaErrorStartupFailureException(kernelName);
			case 10000:
				throw CudaErrorApiFailureBaseException(kernelName);
			default:
				throw CudaGenericKernelException(kernelName, errorCode);
		}
	}
};

struct CuBlasKernelExceptionFactory
{
	static void ThrowException(const std::string& kernelName, const int errorCode)
	{
		switch (errorCode)
		{
			// CudaLightInternalExceptions
			case -3:
			case -2:
			case -1:
				CudaGenericKernelExceptionFactory::ThrowException(kernelName, errorCode);
			case 1:
				throw CuBlasNotInitialisedException(kernelName);
			case 3:
				throw CuBlasAllocFailedException(kernelName);
			case 7:
				throw CuBlasInvalidValueException(kernelName);
			case 8:
				throw CuBlasArchMismatchException(kernelName);
			case 11:
				throw CuBlasMappingErrorException(kernelName);
			case 13:
				throw CuBlasExecutionFailedException(kernelName);
			case 14:
				throw CuBlasInternalErrorException(kernelName);
			case 15:
				throw CuBlasNotSupportedException(kernelName);
			case 16:
				throw CuBlasLicenseErrorException(kernelName);
			default:
				throw CudaGenericKernelException(kernelName, errorCode);
		}
	}
};

struct CuSparseKernelExceptionFactory
{
	static void ThrowException(const std::string& kernelName, const int errorCode)
	{
		switch (errorCode)
		{
			// CudaLightInternalExceptions
			case -3:
			case -2:
			case -1:
				CudaGenericKernelExceptionFactory::ThrowException(kernelName, errorCode);
			case 1:
				throw CuSparseNotInitialisedException(kernelName);
			case 2:
				throw CuSparseAllocFailedException(kernelName);
			case 3:
				throw CuSparseInvalidValueException(kernelName);
			case 4:
				throw CuSparseArchMismatchException(kernelName);
			case 5:
				throw CuSparseMappingErrorException(kernelName);
			case 6:
				throw CuSparseExecutionFailedException(kernelName);
			case 7:
				throw CuSparseInternalErrorException(kernelName);
			case 8:
				throw CuSparseMatrixTypeNotSupportedException(kernelName);
			case 9:
				throw CuSparseZeroPivotException(kernelName);
			default:
				throw CudaGenericKernelException(kernelName, errorCode);
		}
	}
};
