#pragma once
#include <opencv2\opencv.hpp>
#include "DSException.h"
#include "DSRectifier.h"
#include "DSException.h"

class DSFrame{
private:
	cv::Mat left_frame;
	cv::Mat right_frame;

	int width, height;
public:
	DSFrame();
	DSFrame(const cv::String &left_frame_path, const cv::String &right_frame_path);
	DSFrame(const cv::String &left_frame_path, const cv::String &right_frame_path, DSRectifier rectifier);
	DSFrame(cv::Mat left_frame, cv::Mat right_frame);
	DSFrame(cv::Mat left_frame, cv::Mat right_frame, DSRectifier rectifier);

	~DSFrame();

	//Getters
	int get_height(){
		return height;
	}

	int get_width(){
		return width;
	}

	void get_frames(cv::Mat &left_frame, cv::Mat &right_frame){
		left_frame = this->left_frame.clone();
		right_frame = this->right_frame.clone();
	}
};