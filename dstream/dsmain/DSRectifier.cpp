#include "DSRectifier.h"


DSRectifier::DSRectifier(){
}

DSRectifier::DSRectifier(const cv::String params_file){

	cv::FileStorage fs = cv::FileStorage(params_file, cv::FileStorage::READ);
	fs.open(params_file, cv::FileStorage::READ);

	if (!fs.isOpened()) throw DSException(stereo_exceptions::IO_ERROR);

	fs["W"] >> this->width;
	fs["H"] >> this->height;
	
	//Calibrate parameters
	fs["cmL"] >> cmL;
	fs["cmR"] >> cmR;
	fs["dL"] >> dL;
	fs["dR"] >> dR;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["E"] >> E;
	fs["F"] >> F;

	//Rectify parameters
	fs["rL"] >> rL;
	fs["rR"] >> rR;
	fs["pL"] >> pL;
	fs["pR"] >> pR;
	fs["Q"] >> Q;

	fs.release();

	cv::Size im_size = cv::Size(this->width, this->height);

	cv::initUndistortRectifyMap(cmL, dL, rL, pL, im_size, CV_32FC1, left_map_x, left_map_y);
	cv::initUndistortRectifyMap(cmR, dR, rR, pR, im_size, CV_32FC1, right_map_x, right_map_y);
}

DSRectifier::~DSRectifier(){

}

void DSRectifier::rectify(cv::Mat &left_frame, cv::Mat &right_frame){

	cv::Mat left_temp = left_frame.clone();
	cv::Mat right_temp = right_frame.clone();

	if (!(left_temp.rows == right_temp.rows && left_temp.cols == right_temp.cols))
		throw DSException(stereo_exceptions::SIZE_ERROR);
	if (!(left_temp.rows == height && right_temp.rows == height && left_temp.cols == width && right_temp.cols == width))
		throw DSException(stereo_exceptions::SIZE_ERROR);

	remap(left_temp, left_temp, left_map_x, left_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
	remap(right_temp, right_temp, right_map_x, right_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

	left_frame = left_temp.clone();
	right_frame = right_temp.clone();
}