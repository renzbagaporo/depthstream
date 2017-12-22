#pragma once
#include <iostream>
#include <iomanip>

#include "opencv2/opencv.hpp"

#include "DSStream.h"
#include "DSRectifier.h"
#include "DSCalibrator.h"
#include "DSMatcher.h"

#include "support.h"

using namespace cv;
using namespace std;

int main(){

	//// PARAMETERS ////
	int image_width = 1280;
	int image_height = 720;

	int board_width = 7;
	int board_height = 6;
	int images_to_capture = 20;

	float square_size = 5.08f;

	std::vector<cv::Mat> left_images;
	std::vector<cv::Mat> right_images;

	Mat left_image, right_image, left_image_orig, right_image_orig, lr_images;

	//// CALIBRATION ////
	DSStream stream = DSStream(2, 1, image_width, image_height);

	Size board_size = Size(board_width, board_height);

	vector<Point2f> left_corners, right_corners;

	int success = 0, k = 0;

	while (success < images_to_capture)
	{
		if (!stream.read(left_image, right_image)){
			std::cout << "Camera Error." << std::endl;
			break;
		}

		cv::cvtColor(left_image, left_image, CV_BGR2GRAY);
		cv::cvtColor(right_image, right_image, CV_BGR2GRAY);

		left_image_orig = left_image.clone();
		right_image_orig = right_image.clone();
			
		bool left_corners_found = findChessboardCorners(left_image, board_size, left_corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		bool right_corners_found = findChessboardCorners(right_image, board_size, right_corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (left_corners_found)
		{
			cornerSubPix(left_image, left_corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			drawChessboardCorners(left_image, board_size, left_corners, left_corners_found);
		}

		if (right_corners_found)
		{
			cornerSubPix(right_image, right_corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
			drawChessboardCorners(right_image, board_size, right_corners, right_corners_found);
		}

		lr_images = cv::Mat(stream.get_height(), 2 * stream.get_width(), left_image.type());

		//Put im1 on im
		cv::Mat im1_roi(lr_images, cv::Rect(0, 0, left_image.size().width, left_image.size().height));
		left_image.copyTo(im1_roi);

		//Put im2 on im
		cv::Mat im2_roi(lr_images, cv::Rect(right_image.size().width, 0, right_image.size().width, right_image.size().height));
		right_image.copyTo(im2_roi);

		cv::namedWindow("Calibration View");
		imshow("Calibration View", lr_images);

		k = waitKey(10);

		if (left_corners_found && right_corners_found && k == ' ')
		{
			left_images.push_back(left_image_orig);
			right_images.push_back(right_image_orig);

			success++;

			std::cout << "Image " << success << " captured." << std::endl;
		}
	}

	for (int i = 0; i < left_images.size(); i++){
		stringstream ss_l, ss_r;
	
		ss_l << "./ims/high/left/left" << std::setw(3) << std::setfill('0') << i << ".png";
		cv::imwrite(ss_l.str(), left_images[i]);
		ss_r << "./ims/high/right/right" << std::setw(3) << std::setfill('0') << i << ".png";
		cv::imwrite(ss_r.str(), right_images[i]);
	}

	DSCalibrator calibrator = DSCalibrator(left_images, right_images, board_width, board_height, square_size);
	calibrator.write("./ims/high/calibration.yml");

	std::cout << "\n\nRMS Error:" << calibrator.get_rms_error() << "\nReprojection Error: " << calibrator.get_reprojection_error() << std::endl;

	DSRectifier rectifier = DSRectifier("./ims/high/calibration.yml");

	DSStream stream2 = DSStream(2, 1, image_width, image_height, rectifier);

	while (true){

		if (!stream2.read(left_image, right_image)){
			std::cout << "Camera Error." << std::endl;
			break;
		}

		//Put im1 on im
		cv::Mat im1_roi(lr_images, cv::Rect(0, 0, left_image.size().width, left_image.size().height));
		left_image.copyTo(im1_roi);

		//Put im2 on im
		cv::Mat im2_roi(lr_images, cv::Rect(right_image.size().width, 0, right_image.size().width, right_image.size().height));
		right_image.copyTo(im2_roi);

		cv::namedWindow("Calibration View");
		imshow("Calibration View", lr_images);

		cv::waitKey(30);

	}
}