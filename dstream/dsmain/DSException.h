#pragma once
#include <iostream>


enum stereo_exceptions
{
	FORMAT_ERROR,
	IO_ERROR,
	SIZE_ERROR,
	GENERAL_ERROR
};

class DSException : std::exception
{
private:
	stereo_exceptions exception_reason;
public:
	DSException(stereo_exceptions exception_reason);
	~DSException();

	stereo_exceptions get_exception(){
		return exception_reason;
	}
};

