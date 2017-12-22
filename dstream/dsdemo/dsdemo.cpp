#include "dsdemo.h"

#include <iomanip>

#include <opencv2\opencv.hpp>
#include <opencv2\viz.hpp>

#include "DSMatcher.h"
#include "DSFrame.h"


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



dsdemo::dsdemo(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setFixedSize(geometry().width(), geometry().height());
}

dsdemo::~dsdemo()
{

}


void dsdemo::on_depthmap_demo_button_clicked(){
	disable_buttons();

	this->setCursor(Qt::WaitCursor);

	std::vector<DSFrame> frames;
	cv::Mat left, right, disparity, colormap, composite;
	int width = 1242, height = 375;

	for (int i = 0; i <= 153; i++){
		std::stringstream file_number;
		file_number << std::setw(3) << std::setfill('0') << i;
		std::string image_name = "0000000" + file_number.str() + ".png";

		left = cv::imread("./files/sequences/image_00/data/" + image_name, cv::IMREAD_GRAYSCALE);
		cv::resize(left, left, cv::Size(width / 2, height / 2));
		right = cv::imread("./files/sequences/image_01/data/" + image_name, cv::IMREAD_GRAYSCALE);
		cv::resize(right, right, cv::Size(width / 2, height / 2));

		frames.push_back(DSFrame(left, right));
	}

	DSMatcher matcher = DSMatcher(width / 2, height / 2, 128);

	bool loop = true;

	this->hide();
	this->setCursor(Qt::ArrowCursor);

	while (loop){

		for (int i = 0; i <= 153; i++){

			frames[i].get_frames(left, right);

			matcher.compute(frames[i], disparity);

			cv::resize(disparity, disparity, cv::Size(width, height));
			disparity *= 2.0;

			create_colormap(disparity, colormap, 256, 3, 0, cv::COLORMAP_JET);
			cv::resize(left, left, cv::Size(width, height));
			cv::cvtColor(left, left, CV_GRAY2BGR);

			composite = cv::Mat::zeros(cv::Size(width, height * 2), left.type());
			left.copyTo(composite(cv::Rect(0, 0, width, height)));
			colormap.copyTo(composite(cv::Rect(0, height, width, height)));

			cv::imshow("Depthmap Demo", composite);
			char k = cv::waitKey(30);

			if (k == 27){
				loop = false;
				break;
			}
		}
	}

	cv::destroyAllWindows();
	enable_buttons();

	this->show();

}

void dsdemo::on_pointcloud_demo_button_clicked(){
	this->setCursor(Qt::WaitCursor);

	DSRectifier rect = DSRectifier("files/high_calib.yml");

	DSFrame frame = DSFrame("files/pclim/left00.png", "files/pclim/right00.png", rect);
	DSMatcher matcher = DSMatcher(frame.get_width(), frame.get_height(), 128);

	cv::Mat left, right, disp;
	cv::Mat image_3d;
	frame.get_frames(left, right);

	cv::viz::Viz3d viz_window("PointCloud Demo - 3D");

	matcher.compute(frame, disp);
	disp.convertTo(disp, CV_32FC1);
	disp /= 256.0f;
	cv::reprojectImageTo3D(disp, image_3d, rect.Q, true);

	std::vector<cv::Mat> ch;
	cv::split(image_3d, ch);

	cv::Mat mask = cv::Mat::ones(image_3d.size(), CV_8UC1);
	image_3d = image_3d.setTo(0, ch[0] <= -std::numeric_limits<float>::infinity());
	image_3d = image_3d.setTo(0, ch[1] <= -std::numeric_limits<float>::infinity());
	image_3d = image_3d.setTo(0, ch[2] > 100.0);

	this->hide();
	this->setCursor(Qt::ArrowCursor);

	cv::viz::WCloud cloud = cv::viz::WCloud(image_3d, left);
	cv::resize(left, left, cv::Size(left.cols / 2, left.rows / 2));
	while (!viz_window.wasStopped()){
		
		viz_window.showWidget("pointcloud", cloud);
		viz_window.spinOnce(30, true);
		
		cv::imshow("Pointcloud Demo - Image", left);
	}

	cv::destroyAllWindows();
	enable_buttons();

	this->show();
}

void dsdemo::on_tracking_demo_button_clicked(){
	disable_buttons();

	this->setCursor(Qt::WaitCursor);

	cv::Mat left, right, disparity, depthmap, hsv, mask;
	int width = 320, height = 240, disparities = 128;

	DSRectifier rectifier = DSRectifier("files/low_calib.yml");
	DSStream stream = DSStream(2, 1, width, height, rectifier); DSFrame frame;
	DSMatcher matcher = DSMatcher(width, height, disparities);

	cv::Rect rectangle;
	cv::Scalar mean;
	cv::Mat mask_temp, disparity_temp;

	cv::FileStorage fs = cv::FileStorage("files/config.yml", cv::FileStorage::READ);

	if (fs.isOpened()){
		int hue_min, hue_max, sat_min, sat_max, val_min, val_max, morph_size;

		fs["HMIN"] >> hue_min;
		fs["HMAX"] >> hue_max;
		fs["SMIN"] >> sat_min;
		fs["SMAX"] >> sat_max;
		fs["VMIN"] >> val_min;
		fs["VMAX"] >> val_max;
		fs["MSZ"] >> morph_size;

		bool loop = true;

		this->hide();
		this->setCursor(Qt::ArrowCursor);

		while (loop){

			stream.read(frame);
			frame.get_frames(left, right);

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

			cv::resize(left, left, cv::Size(640, 480));

			cv::putText(left, std::to_string(mean[2]) + " cm", cv::Point(0, left.rows - 1), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 3);

			std::cout << mean[2] << std::endl;
			cv::imshow("Tracking Demo", left);

			char k = cv::waitKey(30);
			if (k == 27){
				loop = false;
				break;
			}
		}

		cv::destroyAllWindows();
	}

	enable_buttons();
	this->show();
}

void dsdemo::disable_buttons(){
	ui.tracking_demo_button->setEnabled(false);
	ui.pointcloud_demo_button->setEnabled(false);
	ui.depthmap_demo_button->setEnabled(false);
}

void dsdemo::enable_buttons(){
	ui.tracking_demo_button->setEnabled(true);
	ui.pointcloud_demo_button->setEnabled(true);
	ui.depthmap_demo_button->setEnabled(true);
}