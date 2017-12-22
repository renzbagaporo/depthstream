#pragma once
#include "DSStream.h"

class DSCalibrator
{
	int number_of_images, board_width, board_height;
	int image_height, image_width;
	float square_size;

	double rms_error, reprojection_error;

	//Calibrate matrices
	cv::Mat cmL, cmR, dL, dR, R, T, E, F;

	//Rectify matrices
	cv::Mat rL, rR, pL, pR, Q;

	void cv_calibrate(std::vector<cv::Mat> left_images, std::vector<cv::Mat> right_images);

public:
	DSCalibrator(std::vector<cv::Mat> left_images, std::vector<cv::Mat> right_images, int board_width, int board_height, float square_size);
	~DSCalibrator();

	void write(const cv::String &filename);
	double get_rms_error(){
		return this->rms_error;
	}
	double get_reprojection_error(){
		return this->reprojection_error;
	}
};

