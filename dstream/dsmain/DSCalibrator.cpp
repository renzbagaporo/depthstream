#include "DSCalibrator.h"
#include "DSException.h"

#include <iostream>

using namespace cv;
using namespace std;

DSCalibrator::DSCalibrator(std::vector<cv::Mat> left_images, std::vector<cv::Mat> right_images, int board_width, int board_height, float square_size)
{
	if (left_images.size() != right_images.size()){
		throw new DSException(stereo_exceptions::SIZE_ERROR);
		return;
	}

	this->number_of_images = (int)(left_images.size() + right_images.size()) / 2;

	int temp_board_height = left_images[0].rows, temp_board_width = left_images[0].cols;
	for (int i = 0; i < number_of_images; i++){
		if ((left_images[i].rows != temp_board_height && left_images[i].cols != temp_board_width) &&
			(right_images[i].rows != temp_board_height && right_images[i].cols != temp_board_width)){
			throw new DSException(stereo_exceptions::SIZE_ERROR);
			return;
		}
	}

	this->image_height = temp_board_height;
	this->image_width = temp_board_width;
	this->board_width = board_width;
	this->board_height = board_height;
	this->square_size = square_size;

	cv_calibrate(left_images, right_images);
}

DSCalibrator::~DSCalibrator()
{
}

void DSCalibrator::write(const cv::String &filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);

	fs << "W" << image_width;
	fs << "H" << image_height;

	fs << "cmL" << cmL;	
	fs << "cmR" << cmR;
	fs << "dL" << dL;
	fs << "dR" << dR;
	fs << "R" << R;
	fs << "T" << T;
	fs << "E" << E;
	fs << "F" << F;
	fs << "rL" << rL;
	fs << "rR" << rR;
	fs << "pL" << pL;
	fs << "pR" << pR;
	fs << "Q" << Q;

	fs << "RMSErr" << rms_error;
	fs << "REPROJErr" << reprojection_error;
}

void DSCalibrator::cv_calibrate(std::vector<cv::Mat> left_images, std::vector<cv::Mat> right_images){

	cv::Size boardSize(board_width, board_height);
	const int maxScale = 2;
	const float squareSize = square_size;

	vector<vector<Point2f>> imagePoints[2];
	vector<vector<Point3f>> objectPoints;
	Size imageSize;

	int i, j, k, nimages = number_of_images;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			Mat img;
			
			if (k == 0){
				img = left_images[i];
			}
			else if (k == 1){
				img = right_images[i];
			} else {}
						
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				break;
			}

			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found = findChessboardCorners(timg, boardSize, corners,
					CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}

			if (!found) break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,
				30, 0.01));
		}
		if (k == 2)
		{
			j++;
		}
	}

	this->number_of_images = nimages = j;
	if (nimages < 2)
	{
		throw new DSException(stereo_exceptions::SIZE_ERROR);
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	rms_error = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cmL, dL,
		cmR, dR,
		imageSize, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	stereoRectify(cmL, dL, cmR, dR, cv::Size(image_width, image_height), R, T, rL, rR, pL, pR, Q);
	
	reprojection_error = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (int i = 0; i < number_of_images; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];

		imgpt[0] = Mat(imagePoints[0][i]);
		undistortPoints(imgpt[0], imgpt[0], cmL, dL, Mat(), cmL);
		computeCorrespondEpilines(imgpt[0], 0 + 1, F, lines[0]);

		imgpt[1] = Mat(imagePoints[1][i]);
		undistortPoints(imgpt[1], imgpt[1], cmL, dL, Mat(), cmL);
		computeCorrespondEpilines(imgpt[1], 1 + 1, F, lines[1]);

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			reprojection_error += errij;
		}
		npoints += npt;
	}

	reprojection_error = reprojection_error / (double)npoints;
}