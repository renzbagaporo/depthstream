#pragma once
#include <iostream>

#include <opencv2\viz.hpp>
#include <opencv\highgui.h>

#include "DSStream.h"
#include "DSMatcher.h"
#include "DSFrame.h"
#include "DSRectifier.h"

#include "support.h"

int hue_min = 77, hue_max = 119, sat_min = 77, sat_max = 208, val_min = 45, val_max = 235, morph_size = 15;

void create_mask(cv::Mat hsv, cv::Mat &mask, int hue_min, int hue_max, int sat_min, int sat_max, int val_min, int val_max, int morph_size){
	cv::inRange(hsv, cv::Scalar(hue_min, sat_min, val_min), cv::Scalar(hue_max, sat_max, val_max), mask);

	//morphological opening (remove small objects from the foreground)
	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size)));
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size)));

	//morphological closing (fill small holes in the foreground)
	cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size)));
	cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size)));
}

void create_colormap(cv::Mat disparity, cv::Mat &colormap, int divisor, double max_scale, double min_scale, int mapping){
	colormap = disparity.clone();
	colormap /= divisor;
	colormap.convertTo(colormap, CV_8UC1, max_scale, min_scale);
	
	cv::applyColorMap(colormap, colormap, mapping);
}

void create_depthmap(cv::Mat disparity, cv::Mat q, cv::Mat &depthmap, float divisor){
	depthmap = disparity.clone();

	depthmap.convertTo(depthmap, CV_32FC1);
	depthmap /= divisor;
	cv::reprojectImageTo3D(depthmap, depthmap, q);
}

void create_rectangle(cv::Mat mask, cv::Rect &rectangle, int allowance){
	rectangle = cv::boundingRect(mask);

	rectangle.x = 0;
	rectangle.width = mask.cols;

	rectangle.y = (rectangle.y - allowance < 0) ? 0 : rectangle.y - allowance;
	
	int y_max = rectangle.y + rectangle.height + 2 * allowance;

	if (y_max > mask.rows)
		y_max = mask.rows;

	rectangle.height = abs(y_max - rectangle.y);
}

void on_trackbar(int, void*){
	cv::FileStorage fs = cv::FileStorage("config.yml", cv::FileStorage::WRITE);

	fs << "HMIN" << hue_min;
	fs << "HMAX" << hue_max;
	fs << "SMIN" << sat_min;
	fs << "SMAX" << sat_max;
	fs << "VMIN" << val_min;
	fs << "VMAX" << val_max;
	fs << "MSZ" << morph_size;
}

int main(){

	int state = 2;

	int width = 320, height = 240, disparities = 128;

	DSRectifier rectifier = DSRectifier("calibration.yml");
	DSStream stream = DSStream(2, 1, width, height, rectifier);

	DSMatcher matcher = DSMatcher(width, height, disparities);

	DSFrame frame;
	cv::Mat left, right, disparity, colormap, depthmap, hsv, mask;

	cv::FileStorage fs = cv::FileStorage("config.yml", cv::FileStorage::READ);

	if (fs.isOpened()){
		fs["HMIN"] >> hue_min;
		fs["HMAX"] >> hue_max;
		fs["SMIN"] >> sat_min;
		fs["SMAX"] >> sat_max;
		fs["VMIN"] >> val_min;
		fs["VMAX"] >> val_max;
		fs["MSZ"] >> morph_size;
	}
	
	bool iterate = true;

	cv::Rect rectangle;
	cv::Scalar mean;
	cv::Mat mask_temp, disparity_temp;

	while (iterate){
		stream.read(frame);
		frame.get_frames(left, right);

		char k = cv::waitKey(30);
		
		if (k == 27){
			break;
		}
		else if (k == 's'){
			state++;
			if (state > 2) state = 0;
			cv::destroyAllWindows();
		} else{}
		
		switch (state)
		{
		case 0:
		{
			cv::namedWindow("Controls");

			cv::createTrackbar("hue_min_trackbar", "Controls", &hue_min, 179, on_trackbar); //Hue (0 - 179)
			cv::createTrackbar("hue_max_trackbar", "Controls", &hue_max, 179, on_trackbar);

			cv::createTrackbar("sat_min_trackbar", "Controls", &sat_min, 255, on_trackbar); //Saturation (0 - 255)
			cv::createTrackbar("sat_max_trackbar", "Controls", &sat_max, 255, on_trackbar);

			cv::createTrackbar("val_min_trackbar", "Controls", &val_min, 255, on_trackbar); //Value (0 - 255)
			cv::createTrackbar("val_max_trackbar", "Controls", &val_max, 255, on_trackbar);

			cv::createTrackbar("morph_size_trackbar", "Controls", &morph_size, 255, on_trackbar);

			cv::cvtColor(left, hsv, CV_BGR2HSV);
			create_mask(hsv, mask, hue_min, hue_max, sat_min, sat_max, val_min, val_max, morph_size);
			cv::imshow("Mask", mask);
		}
			break;
		case 1:
		{
			//Create an image mask
			cv::cvtColor(left, hsv, CV_BGR2HSV);
			create_mask(hsv, mask, hue_min, hue_max, sat_min, sat_max, val_min, val_max, morph_size);

			//Find the bounding rectangle
			create_rectangle(mask, rectangle, 20);

			//Compute the disparity
			matcher.compute(DSFrame(left, right), rectangle, disparity, 8, 17, 34, 15, 6, 4, 1);
			disparity.copyTo(disparity, mask);

			//Compute the mean disparity
			mean = cv::mean(disparity, mask);
			
			//Create a new mask. If the disparity is less than a certain percent of the mean, disable the masking pixel
			disparity.convertTo(disparity_temp, CV_32FC1);
			cv::threshold(disparity_temp, mask_temp, 0.9f * mean[0], 255, cv::THRESH_BINARY);
			mask_temp.convertTo(mask_temp, CV_8UC1);
			mask_temp = mask & mask_temp;

			//Create a depth map and find the mean
			create_depthmap(disparity, rectifier.Q, depthmap, 256);
			mean = cv::mean(depthmap, mask_temp);

			//Draw bounding rectangle
			rectangle = cv::boundingRect(mask_temp);
			cv::rectangle(left, rectangle, cv::Scalar(0, 0, 255));

			std::cout << mean[2] << std::endl;
			cv::imshow("Image", left);
		}
			break;
		case 2:
		{
			matcher.compute(DSFrame(left, right), disparity, 8, 17, 34, 15, 6, 4, 1);
			create_colormap(disparity, colormap, 256, 4, 0, cv::COLORMAP_JET);
			cv::imshow("Colormap", colormap);
		}
			break;
		default:
			break;
		}
	}

	return 0;
}