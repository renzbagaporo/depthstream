#include "DSMatcher.h"
//#define TIME

#ifdef TIME
#include <arrayfire.h>
#endif

DSMatcher::DSMatcher(){}

DSMatcher::DSMatcher(int width, int height, int disparities)
{
	//Setup parameters
	this->width = width;
	this->height = height;
	this->disparities = disparities;

	//Initalize core
	core.setup(this->width, this->height, this->disparities);
}

DSMatcher::~DSMatcher(){

}

bool DSMatcher::compute(DSFrame frame, cv::Mat &disp_im, int gamma, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int region_voting_iterations, int disparity_tolerance){
#ifdef TIME
	af::timer::start();
#endif


	float ad_gamma = gamma / 100.0f;
	float census_gamma = 1.0f - (ad_gamma);
	
	if (!(frame.get_width() == width && frame.get_height() == height)) throw DSException(stereo_exceptions::SIZE_ERROR);

	cv::Mat left_frame, right_frame;
	frame.get_frames(left_frame, right_frame);

	if (left_frame.channels() == 3) cv::cvtColor(left_frame, left_frame, CV_BGR2GRAY);
	if (right_frame.channels() == 3) cv::cvtColor(right_frame, right_frame, CV_BGR2GRAY);

	//Transfer images to device
	core.copy_from_host_to_device(left_frame.data, DSCore::core_data::LEFT_DATA);
	core.copy_from_host_to_device(right_frame.data, DSCore::core_data::RIGHT_DATA);

	//Compute
	core.stereo_match(arm_length, max_arm_length, arm_threshold, strict_arm_threshold, ad_gamma, census_gamma, disparity_tolerance, region_voting_iterations);

	//Transfer result to host
	cv::Mat disparity_temp = cv::Mat::zeros(frame.get_height(), frame.get_width(), CV_16UC1);
	core.copy_from_device_to_host(disparity_temp.data, DSCore::core_data::FINAL_DISP_DATA);

	disp_im = disparity_temp.clone();

#ifdef TIME
	printf("elapsed seconds: %g\n", af::timer::stop());
#endif
	return true;
}

bool DSMatcher::compute(DSFrame frame, cv::Rect roi, cv::Mat &disp_im, int gamma, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int region_voting_iterations, int disparity_tolerance){
#ifdef TIME
	af::timer::start();
#endif

	int w = roi.width;
	int h = roi.height;

	if (w > width) throw new DSException(stereo_exceptions::SIZE_ERROR);
	if (h > height) throw new DSException(stereo_exceptions::SIZE_ERROR);

	float ad_gamma = gamma / 100.0f;
	float census_gamma = 1.0f - (ad_gamma);

	if (!(frame.get_width() == width && frame.get_height() == height)) throw DSException(stereo_exceptions::SIZE_ERROR);

	cv::Mat left_frame, right_frame;
	frame.get_frames(left_frame, right_frame);

	if (left_frame.channels() == 3) cv::cvtColor(left_frame, left_frame, CV_BGR2GRAY);
	if (right_frame.channels() == 3) cv::cvtColor(right_frame, right_frame, CV_BGR2GRAY);

	cv::Mat left_temp = cv::Mat::zeros(frame.get_height(), frame.get_width(), CV_8UC1);
	cv::Mat right_temp = cv::Mat::zeros(frame.get_height(), frame.get_width(), CV_8UC1);

	left_frame(roi).copyTo(left_temp(cv::Rect(0, 0, roi.width, roi.height)));
	right_frame(roi).copyTo(right_temp(cv::Rect(0, 0, roi.width, roi.height)));

	left_frame = left_temp;
	right_frame = right_temp;

	//Transfer images to device
	core.copy_from_host_to_device(left_frame.data, DSCore::core_data::LEFT_DATA);
	core.copy_from_host_to_device(right_frame.data, DSCore::core_data::RIGHT_DATA);

	//Compute
	core.stereo_match(arm_length, max_arm_length, arm_threshold, strict_arm_threshold, ad_gamma, census_gamma, disparity_tolerance, region_voting_iterations, w, h);

	//Transfer result to host
	cv::Mat disparity_temp = cv::Mat::zeros(frame.get_height(), frame.get_width(), CV_16UC1);
	core.copy_from_device_to_host(disparity_temp.data, DSCore::core_data::FINAL_DISP_DATA);

	disp_im = cv::Mat::zeros(frame.get_height(), frame.get_width(), CV_16UC1);

	disparity_temp(cv::Rect(0, 0, roi.width, roi.height)).copyTo(disp_im(roi));

#ifdef TIME
	printf("elapsed seconds: %g\n", af::timer::stop());
#endif
	return true;
}

