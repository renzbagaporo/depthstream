#pragma once
#include <opencv2\opencv.hpp>
#include "DSException.h"

class DSRectifier{
private:

	cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;
	int width, height;

public:

	//Calibrate parameters
	cv::Mat cmL;
	cv::Mat cmR;
	cv::Mat dL;
	cv::Mat dR;
	cv::Mat R;
	cv::Mat T;
	cv::Mat E;
	cv::Mat F;

	//Rectify parameters
	cv::Mat rL;
	cv::Mat rR;
	cv::Mat pL;
	cv::Mat pR;
	cv::Mat Q;

	DSRectifier();
	DSRectifier(const cv::String params_file);
	~DSRectifier();

	void rectify(cv::Mat &left_frame, cv::Mat &right_frame);

	int get_height(){
		return height;
	}
	int get_width(){
		return width;
	}
};