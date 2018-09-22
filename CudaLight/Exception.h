#pragma once

#include <exception>
#include <string>

class Exception : public std::exception
{
public:
	Exception(const std::string& message = "") : message(message) {}
	Exception(const char* message = "") : message(message) {}
	~Exception() = default;

	Exception(const Exception& rhs) = default;
	Exception(Exception&& rhs) = default;
	Exception& operator=(const Exception& rhs) = default;
	Exception& operator=(Exception&& rhs) = default;

	char const* what() const override final
	{
		return message.c_str();
	}

protected:
	const std::string message;
};

class InternalErrorException : public Exception
{
public:
	InternalErrorException(const std::string& message = "")
		: Exception("InternalErrorException: " + message)
	{
	}
};

class ExpectedEvenSizeException : public Exception
{
public:
	ExpectedEvenSizeException(const std::string& message = "")
		: Exception("ExpectedEvenSizeException: " + message)
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