#pragma once

#include <exception>
#include <string>

class Exception : public std::exception
{
public:
    explicit Exception(const std::string& message = "") : _message(message) {}
    explicit Exception(const char* message = "") : _message(message) {}
	virtual ~Exception() override = default;

	Exception(const Exception& rhs) = default;
	Exception(Exception&& rhs) = default;
	Exception& operator=(const Exception& rhs) = delete;
	Exception& operator=(Exception&& rhs) = delete;

	char const* what() const noexcept final
	{
		return _message.c_str();
	}

protected:
	const std::string _message;
};

class InternalErrorException : public Exception
{
public:
	explicit InternalErrorException(const std::string& message = "")
		: Exception("InternalErrorException: " + message)
	{
	}
};

class ExpectedEvenSizeException : public Exception
{
public:
    explicit ExpectedEvenSizeException(const std::string& message = "")
		: Exception("ExpectedEvenSizeException: " + message)
	{
	}
};

class NotSupportedException : public Exception
{
public:
    explicit NotSupportedException(const std::string& message = "")
		: Exception("NotSupportedException: " + message)
	{
	}
};

class NotImplementedException : public Exception
{
public:
    explicit NotImplementedException(const std::string& message = "")
		: Exception("NotImplementedException: " + message)
	{
	}
};

class BufferNotInitialisedException : public Exception
{
public:
    explicit BufferNotInitialisedException(const std::string& message = "")
		: Exception("BufferNotInitialisedException: " + message)
	{
	}
};
