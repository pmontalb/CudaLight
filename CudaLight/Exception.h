#pragma once

#include <exception>
#include <string>

class Exception : std::exception
{
public:
	Exception(const std::string& message = "") : message(message) {}
	Exception(const char* message = "") : message(message) {}
	~Exception() = default;

	Exception(const Exception& rhs) = default;
	Exception(Exception&& rhs) = default;
	Exception& operator=(const Exception& rhs) = default;
	Exception& operator=(Exception&& rhs) = default;

	char const* what() const override
	{
		return message.c_str();
	}

protected:
	const std::string message;
};

class CudaKernelException : public Exception
{
public:
	CudaKernelException(const std::string& kernelName, const int errorCode = -1)
		: Exception(kernelName + " returned " + std::to_string(errorCode))
	{
	}
};

class NotSupportedException : public Exception
{
public:
	NotSupportedException(const std::string& message = "")
		: Exception("NotSupportedException: " + message)
	{
	}
};

class NotImplementedException : public Exception
{
public:
	NotImplementedException(const std::string& message = "")
		: Exception("NotImplementedException: " + message)
	{
	}
};

class BufferNotInitialisedException : public Exception
{
public:
	BufferNotInitialisedException(const std::string& message = "")
		: Exception("BufferNotInitialisedException: " + message)
	{
	}
};