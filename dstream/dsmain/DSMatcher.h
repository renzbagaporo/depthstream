#pragma once
#include <opencv2\opencv.hpp>

#include "DSStream.h"
#include "DSException.h"
#include "DSProcess.h"
#include "DSFrame.h"
#include "DSCore.h"

//Class Declaration
class DSMatcher
{
private:
	//Core
	DSCore core;

	//Stereo parameters
	int width, height, disparities;

public:
	DSMatcher();
	DSMatcher::DSMatcher(int width, int height, int disparities);
	~DSMatcher();

	//Class methods
	bool compute(DSFrame frame, cv::Mat &disp_im, int gamma = 30, int arm_length = 8, int max_arm_length = 17, 
		int arm_threshold = 15, int strict_arm_threshold = 6, int region_voting_iterations = 4, int disparity_tolerance = 1);
	bool compute(DSFrame frame, cv::Rect roi, cv::Mat &disp_im, int gamma = 30, int arm_length = 8, int max_arm_length = 17,
		int arm_threshold = 15, int strict_arm_threshold = 6, int region_voting_iterations = 4, int disparity_tolerance = 1);
};

