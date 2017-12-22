#include <opencv2\opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

void show(cv::Mat im){
	cv::imshow("", im);
	cv::waitKey(0);
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

	cv::imshow("", im);
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

	cv::imshow("", adjMap);
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

		cv::imshow("", adjMap);
		cv::waitKey(0);
	}
}


