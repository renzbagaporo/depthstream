#pragma once
#include <opencv2\opencv.hpp>
#include "DSException.h"
#include "DSRectifier.h"
#include "DSFrame.h"

enum stream_type{FILE_STREAM, DEVICE_STREAM};

class  DSStream
{
private:
	cv::VideoCapture left_capture;
	cv::VideoCapture right_capture;
	
	stream_type type;

	DSRectifier rectifier;
	bool should_rectify;

	cv::Mat left_frame, right_frame;
	
	int height, width;

public:
	DSStream();

	DSStream(const cv::String &left_video_path, const cv::String &right_video_path);
	DSStream(const cv::String &left_video_path, const cv::String &right_video_path, DSRectifier rectifier);
	DSStream(int left_device_id, int right_device_id, int width, int height);
	DSStream(int left_device_id, int right_device_id, int width, int height, DSRectifier rectifier);

	~DSStream();

	//Class methods
	bool read(cv::Mat &left_frame, cv::Mat &right_frame);
	bool DSStream::read(DSFrame &frame);
	void reset();

	//Getters
	int get_width(){
		return width;
	}

	int get_height(){
		return height;
	}

	stream_type get_stream_type(){ return type; }
};

