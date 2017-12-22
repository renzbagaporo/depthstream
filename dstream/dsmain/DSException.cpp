#include "DSException.h"

DSException::DSException(stereo_exceptions exception_reason)
{
	this->exception_reason = exception_reason;

}


DSException::~DSException()
{
}


