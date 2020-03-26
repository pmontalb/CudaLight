#pragma once

#include <exception>

namespace cl
{
	class Exception: public std::exception
	{
	};

	class NotImplementedException: public Exception
	{
		inline const char* what() const noexcept final { return "NotImplemented"; }
	};

	class MklException: public Exception
	{
	public:
		explicit MklException(const char* callerFunction)
			: _callerFunction(callerFunction)
		{

		}
		MklException(const MklException& rhs) = default;
		MklException& operator=(const MklException& rhs) = default;
		inline const char* what() const noexcept final { return _callerFunction; }

	private:
		const char* _callerFunction;
	};

	class OpenBlasException: public Exception
	{
	public:
		explicit OpenBlasException(const char* callerFunction)
				: _callerFunction(callerFunction)
		{

		}
		OpenBlasException(const OpenBlasException& rhs) = default;
		OpenBlasException& operator=(const OpenBlasException& rhs) = default;
		inline const char* what() const noexcept final { return _callerFunction; }

	private:
		const char* _callerFunction;
	};
}
