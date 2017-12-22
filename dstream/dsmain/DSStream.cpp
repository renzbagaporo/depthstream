#include "DSStream.h"


DSStream::DSStream(){}

DSStream::~DSStream(){
	left_capture.release();
	right_capture.release();
}

DSStream::DSStream(const cv::String &left_video_path, const cv::String &right_video_path){
	this->left_capture = cv::VideoCapture(left_video_path);
	this->right_capture = cv::VideoCapture(right_video_path);

	if (!(left_capture.isOpened() && right_capture.isOpened())) throw DSException(stereo_exceptions::IO_ERROR);
	
	this->type = stream_type::FILE_STREAM;

	if (!((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) == (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH) 
		&& (int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) == (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT))) 
		throw DSException(stereo_exceptions::SIZE_ERROR);

	this->width = ((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) + (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH)) / 2;
	this->height = ((int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) + (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT)) / 2;

	should_rectify = false;
}

DSStream::DSStream(const cv::String &left_video_path, const cv::String &right_video_path, DSRectifier rectifier){
	this->left_capture = cv::VideoCapture(left_video_path);
	this->right_capture = cv::VideoCapture(right_video_path);

	if (!(left_capture.isOpened() && right_capture.isOpened())) throw DSException(stereo_exceptions::IO_ERROR);
	this->type = stream_type::FILE_STREAM;


	if (!((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) == (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH)
		&& (int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) == (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT)))
		throw DSException(stereo_exceptions::SIZE_ERROR);

	this->width = ((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) + (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH)) / 2;
	this->height = ((int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) + (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT)) / 2;

	if (!(this->width == rectifier.get_width() && this->height == rectifier.get_height())) throw DSException(stereo_exceptions::SIZE_ERROR);

	should_rectify = true;
}

DSStream::DSStream(int left_device_id, int right_device_id, int width, int height){
	this->left_capture = cv::VideoCapture(left_device_id);
	this->right_capture = cv::VideoCapture(right_device_id);
	this->type = stream_type::DEVICE_STREAM;

	left_capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	left_capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

	right_capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	right_capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

	this->width = ((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) + (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH)) / 2;
	this->height = ((int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) + (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT)) / 2;

	should_rectify = false;
}

DSStream::DSStream(int left_device_id, int right_device_id, int width, int height, DSRectifier rectifier){
	this->left_capture = cv::VideoCapture(left_device_id);
	this->right_capture = cv::VideoCapture(right_device_id);
	this->type = stream_type::DEVICE_STREAM;

	left_capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	right_capture.set(CV_CAP_PROP_FRAME_WIDTH, width);

	left_capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	right_capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

	this->width = ((int)left_capture.get(CV_CAP_PROP_FRAME_WIDTH) + (int)right_capture.get(CV_CAP_PROP_FRAME_WIDTH)) / 2;
	this->height = ((int)left_capture.get(CV_CAP_PROP_FRAME_HEIGHT) + (int)right_capture.get(CV_CAP_PROP_FRAME_HEIGHT)) / 2;;

	if (!(this->width == rectifier.get_width() && this->height == rectifier.get_height())) throw DSException(stereo_exceptions::SIZE_ERROR);

	this->rectifier = rectifier;

	should_rectify = true;
}

bool DSStream::read(cv::Mat &left_frame, cv::Mat &right_frame){
	if (left_capture.grab() && right_capture.grab()){
		bool read_success = left_capture.retrieve(this->left_frame) && right_capture.retrieve(this->right_frame);

		if (read_success){

			if (should_rectify) this->rectifier.rectify(this->left_frame, this->right_frame);

			left_frame = this->left_frame.clone();
			right_frame = this->right_frame.clone();
		}
		return read_success;
	}
	return false;
}

bool DSStream::read(DSFrame &frame){
	if (left_capture.grab() && right_capture.grab()){
		bool read_success = left_capture.retrieve(this->left_frame) && right_capture.retrieve(this->right_frame);

		if (read_success){
			if (should_rectify) this->rectifier.rectify(this->left_frame, this->right_frame);
			frame = DSFrame(this->left_frame.clone(), this->right_frame.clone());
		}
		return read_success;
	}
	return false;
}


void DSStream::reset(){
	left_capture.set(CV_CAP_PROP_POS_AVI_RATIO, 0.0);
	right_capture.set(CV_CAP_PROP_POS_AVI_RATIO, 0.0);
}