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
}