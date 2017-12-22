#pragma once
#include <opencv2\opencv.hpp>

void show(cv::Mat im, int wait){
	cv::namedWindow("ims", CV_WINDOW_AUTOSIZE);
	cv::imshow("ims", im);
	cv::waitKey(wait);
}

void show(cv::Mat im){
	cv::namedWindow("ims", CV_WINDOW_AUTOSIZE);
	cv::imshow("ims", im);
	cv::waitKey(0);
}

void create_colormap(cv::Mat disparity, cv::Mat &colormap, int divisor, double max_scale, double min_scale, int mapping){
	colormap = disparity.clone();
	colormap /= divisor;
	colormap.convertTo(colormap, CV_8UC1, max_scale, min_scale);

	cv::applyColorMap(colormap, colormap, mapping);
}

void show(cv::Mat im1, cv::Mat im2){
	int height = (im1.size().height > im2.size().height) ? im1.size().height : im2.size().height;
	int width = im1.size().width + im2.size().width;
	cv::Mat im = cv::Mat(height, width, im1.type());

	//Put im1 on im
	cv::Mat im1_roi(im, cv::Rect(0, 0, im1.size().width, im2.size().height));
	im1.copyTo(im1_roi);

	//Put im2 on im
	cv::Mat im2_roi(im, cv::Rect(im1.size().width, 0, im2.size().width, im2.size().height));
	im2.copyTo(im2_roi);

	cv::namedWindow("ims", CV_WINDOW_NORMAL);

	cv::imshow("ims", im);
}

void showsc(cv::Mat im1, cv::Mat im2){
	double min1;
	double max1;
	cv::minMaxIdx(im1, &min1, &max1);
	cv::Mat adjMap1;

	// Histogram Equalization
	max1 = 48;
	min1 = 0;
	float scale1 = (float)(255 / (max1 - min1));
	im1.convertTo(adjMap1, CV_8UC1, scale1, -min1*scale1);

	double min2;
	double max2;
	cv::minMaxIdx(im2, &min2, &max2);
	cv::Mat adjMap2;

	// Histogram Equalization
	max2 = 48;
	min2 = 0;
	float scale2 = (float)(255 / (max2 - min2));
	im2.convertTo(adjMap2, CV_8UC1, scale2, -min2*scale2);

	show(adjMap1, adjMap2);
	cv::waitKey(0);
}

void showsc(cv::Mat im1, cv::Mat im2, double min, double max){
	double min1;
	double max1;
	cv::Mat adjMap1;

	// Histogram Equalization
	max1 = max;
	min1 = min;
	float scale1 = (float)(255 / (max1 - min1));
	im1.convertTo(adjMap1, CV_8UC1, scale1, -min1*scale1);

	double min2;
	double max2;
	cv::Mat adjMap2;

	// Histogram Equalization
	max2 = max;
	min2 = min;
	float scale2 = (float)(255 / (max2 - min2));
	im2.convertTo(adjMap2, CV_8UC1, scale2, -min2*scale2);

	show(adjMap1, adjMap2);
	cv::waitKey(0);
}

void showsc(cv::Mat im){
	double min;
	double max;
	cv::minMaxIdx(im, &min, &max);
	cv::Mat adjMap;

	// Histogram Equalization
	float scale = (float)(255 / (max - min));
	im.convertTo(adjMap, CV_8UC1, scale, -min*scale);

	show(adjMap);
}

void showsc(cv::Mat im, int wait){
	double min;
	double max;
	cv::minMaxIdx(im, &min, &max);
	cv::Mat adjMap;

	// Histogram Equalization
	float scale = (float)(255 / (max - min));
	im.convertTo(adjMap, CV_8UC1, scale, -min*scale);

	show(adjMap, wait);
}


void showsc(cv::Mat im, double min, double max){
	cv::Mat adjMap;

	// Histogram Equalization
	float scale = (float)(255 / (max - min));
	im.convertTo(adjMap, CV_8UC1, scale, -min*scale);

	show(adjMap);
	cv::waitKey(0);
}

void showsc(std::vector<cv::Mat> ims){
	double min;
	double max;

	// Histogram Equalization
	for (int i = 0; i < ims.size(); i++){
		cv::minMaxIdx(ims[i], &min, &max);
		cv::Mat adjMap;
		float scale = (float)(255 / (max - min));
		ims[i].convertTo(adjMap, CV_8UC1, scale, -min*scale);

		show(adjMap);
		cv::waitKey(0);
	}
}


