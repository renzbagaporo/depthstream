#include "DSFrame.h"

DSFrame::DSFrame(){}
DSFrame::~DSFrame(){}

DSFrame::DSFrame(const cv::String &left_frame_path, const cv::String &right_frame_path){
	this->left_frame = cv::imread(left_frame_path);
	this->right_frame = cv::imread(right_frame_path);

	if (!(left_frame.data && right_frame.data)) throw DSException(stereo_exceptions::IO_ERROR);

	if (!(this->left_frame.rows == this->right_frame.rows && this->left_frame.cols == this->right_frame.cols)) throw DSException(stereo_exceptions::SIZE_ERROR);

	this->height = (this->left_frame.rows + this->right_frame.rows) / 2;
	this->width = (this->left_frame.cols + this->right_frame.cols) / 2;
}

DSFrame::DSFrame(const cv::String &left_frame_path, const cv::String &right_frame_path, DSRectifier rectifier){
	this->left_frame = cv::imread(left_frame_path);
	this->right_frame = cv::imread(right_frame_path);

	if (!(left_frame.data && right_frame.data)) throw DSException(stereo_exceptions::IO_ERROR);

	if (!(this->left_frame.rows == this->right_frame.rows && this->left_frame.cols == this->right_frame.cols)) throw DSException(stereo_exceptions::SIZE_ERROR);

	this->height = (this->left_frame.rows + this->right_frame.rows) / 2;
	this->width = (this->left_frame.cols + this->right_frame.cols) / 2;

	if (!(this->height == rectifier.get_height() && this->width == rectifier.get_width())) throw DSException(stereo_exceptions::SIZE_ERROR);

	rectifier.rectify(this->left_frame, this->right_frame);
}

DSFrame::DSFrame(cv::Mat left_frame, cv::Mat right_frame){
	this->left_frame = left_frame.clone();
	this->right_frame = right_frame.clone();

	if (!(this->left_frame.data && this->right_frame.data)) throw DSException(stereo_exceptions::IO_ERROR);

	if (!(this->left_frame.rows == this->right_frame.rows && this->left_frame.cols == this->right_frame.cols)) throw DSException(stereo_exceptions::SIZE_ERROR);

	this->height = (this->left_frame.rows + this->right_frame.rows) / 2;
	this->width = (this->left_frame.cols + this->right_frame.cols) / 2;
}

DSFrame::DSFrame(cv::Mat left_frame, cv::Mat right_frame, DSRectifier rectifier){
	this->left_frame = left_frame.clone();
	this->right_frame = right_frame.clone();

	if (!(this->left_frame.data && this->right_frame.data)) throw DSException(stereo_exceptions::IO_ERROR);
	
	if (!(this->left_frame.rows == this->right_frame.rows && this->left_frame.cols == this->right_frame.cols)) throw DSException(stereo_exceptions::SIZE_ERROR);

	this->height = (this->left_frame.rows + this->right_frame.rows) / 2;
	this->width = (this->left_frame.cols + this->right_frame.cols) / 2;

	if (!(this->height == rectifier.get_height() && this->width == rectifier.get_width())) throw DSException(stereo_exceptions::SIZE_ERROR);

	rectifier.rectify(this->left_frame, this->right_frame);
}