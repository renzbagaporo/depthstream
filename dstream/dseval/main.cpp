#pragma once
#include <fstream>
#include <iomanip>
#include <chrono>

#include "opencv2\opencv.hpp"
#include "opencv2\cudastereo.hpp"

#include "DSMatcher.h"
#include "DSFrame.h"

#include "vx_bm.hpp"
#include "vx_sgbm.hpp"

#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>
#include <NVXIO/Utility.hpp>


#define ALGO=BM

std::string float2str(float f){
	std::stringstream s;
	s << f;
	return s.str();
}

static void VX_CALLBACK myLogCallback(vx_context /*context*/, vx_reference /*ref*/, vx_status /*status*/, const vx_char string[])
{
	std::cout << "VisionWorks LOG : " << string << std::endl;
}

int main(){

	//Read KITTI Dataset
	std::vector<cv::Mat> right_images;
	std::vector<cv::Mat> left_images;

	std::vector<std::string> image_names;

	for (int i = 0; i <= 193; i++){
		//Read images
		std::stringstream file_number;
		file_number << std::setw(3) << std::setfill('0') << i;
		std::string image_name = "000" + file_number.str() + "_10.png";
		image_names.push_back(image_name);

		left_images.push_back(cv::imread("../kitteval/data_stereo_flow/training/image_0/" + image_name, cv::IMREAD_GRAYSCALE));
		right_images.push_back(cv::imread("../kitteval/data_stereo_flow/training/image_1/" + image_name, cv::IMREAD_GRAYSCALE));
	}

#if ALGO == OCV_BM
	//BM
	for (int x = 0; x < 2; x++)
	{
		cv::Ptr<cv::StereoBM> bm;
		bm = cv::StereoBM::create(256, 9);
		bm->setPreFilterSize(41);
		bm->setPreFilterCap(31);
		bm->setTextureThreshold(20);
		bm->setUniquenessRatio(10);
		bm->setSpeckleWindowSize(100);
		bm->setSpeckleRange(32);

		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		for (int i = 0; i <= 193; i++){
			cv::Mat disparity;

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			bm->compute(left_images[i], right_images[i], disparity);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			disparity = disparity.setTo(0, disparity <= 0);
			disparity.convertTo(disparity, CV_16UC1);
			disparity *= 16;

			disparities.push_back(disparity);
			times.push_back((float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());


			std::cout << "BM" << i << std::endl;
		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/OCVBM/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());
		}
	}

#elif ALGO == OCV_SGBM
	//SGBM
	for (int x = 0; x < 2; x++)
	{
		cv::Ptr<cv::StereoSGBM> sgbm;
		sgbm = cv::StereoSGBM::create(0, 256, 3);
		sgbm->setPreFilterCap(63);
		sgbm->setP1(36);
		sgbm->setP2(288);
		sgbm->setMinDisparity(0);
		sgbm->setNumDisparities(256);
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setSpeckleRange(32);
		sgbm->setDisp12MaxDiff(1);
		sgbm->setMode(cv::StereoSGBM::MODE_HH);

		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		for (int i = 0; i <= 193; i++){
			cv::Mat disparity;

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			sgbm->compute(left_images[i], right_images[i], disparity);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			disparity = disparity.setTo(0, disparity <= 0);
			disparity.convertTo(disparity, CV_16UC1);
			disparity *= 16;
			
			disparities.push_back(disparity);
			times.push_back((float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());


			std::cout << "SGBM" << i << std::endl;

		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/OCVSGBM/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());

		}
	}
#elif ALGO == DSTREAM 
	//DSTREAM
	for (int x = 0; x < 2; x++)
	{
		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		for (int i = 0; i <= 193; i++){
			DSMatcher dstream = DSMatcher(left_images[i].cols, left_images[i].rows, 256);
			DSFrame frame = DSFrame(left_images[i], right_images[i]);

			cv::Mat disparity;

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			dstream.compute(frame, disparity, 30, 8, 17, 15, 6, 1, 1);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			disparities.push_back(disparity);
			times.push_back((float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());


			std::cout << "DSTREAM" << i << std::endl;

		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/DSTREAM/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());

		}
	}
#elif ALGO == GPUOCV_BM
	//BM-GPU
	for (int x = 0; x < 2; x++)
	{
		cv::Ptr<cv::cuda::StereoBM> bm_gpu;
		bm_gpu = cv::cuda::createStereoBM(256, 9);
		bm_gpu->setPreFilterCap(31);
		bm_gpu->setMinDisparity(0);
		bm_gpu->setTextureThreshold(10);
		bm_gpu->setUniquenessRatio(15);
		bm_gpu->setSpeckleWindowSize(100);
		bm_gpu->setSpeckleRange(32);

		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		cv::Mat disparity; cv::cuda::GpuMat gpu_result;
		cv::cuda::GpuMat left_im;
		cv::cuda::GpuMat right_im;

		for (int i = 0; i <= 193; i++){

			left_im = cv::cuda::GpuMat(left_images[i]);
			right_im = cv::cuda::GpuMat(right_images[i]);

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			bm_gpu->compute(left_im, right_im, gpu_result);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			disparity = cv::Mat(gpu_result);
			
			disparity = disparity.setTo(0, disparity <= 0);
			disparity.convertTo(disparity, CV_16UC1);
			disparity *= 256;

			disparities.push_back(disparity);
			times.push_back((float)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());


			std::cout << "OCVBMGPU" << i << std::endl;
		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/OCVBMGPU/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());
		}
	}
#elif ALGO == GPUNVX_BM
	for (int x = 0; x < 2; x++)
	{ // NVXBM
		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		for (int i = 0; i <= 193; i++){
			{
				nvxio::ContextGuard context;
				vxRegisterLogCallback(context, &myLogCallback, vx_false_e);

				cv::Size size = left_images[i].size();

				//Import left image to VX context 
				vx_imagepatch_addressing_t left_addr;
				left_addr.dim_x = vx_uint32(size.width);
				left_addr.dim_y = vx_uint32(size.height);
				left_addr.stride_x = vx_int32(sizeof(vx_uint8));
				left_addr.stride_y = vx_int32(left_images[i].step);

				void *left_ptrs[] = { left_images[i].data };

				vx_image left_vx = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &left_addr, left_ptrs, VX_IMPORT_TYPE_HOST);
				NVXIO_CHECK_REFERENCE(left_vx);

				//Import right image to VX context
				vx_imagepatch_addressing_t right_addr;
				right_addr.dim_x = vx_uint32(size.width);
				right_addr.dim_y = vx_uint32(size.height);
				right_addr.stride_x = vx_int32(sizeof(vx_uint8));
				right_addr.stride_y = vx_int32(right_images[i].step);

				void *right_ptrs[] = { right_images[i].data };

				vx_image right_vx = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &right_addr, right_ptrs, VX_IMPORT_TYPE_HOST);
				NVXIO_CHECK_REFERENCE(right_vx);

				//Create output image
				cv::Mat disparity = cv::Mat::zeros(size, CV_8UC1);

				vx_image disparity_vx = vxCreateImage(context, size.width, size.height, VX_DF_IMAGE_U8);
				vx_imagepatch_addressing_t disparity_addr;
				disparity_addr.dim_x = vx_uint32(size.width);
				disparity_addr.dim_y = vx_uint32(size.height);
				disparity_addr.stride_x = vx_int32(sizeof(vx_uint8));
				disparity_addr.stride_y = vx_int32(disparity.step);

				NVXIO_CHECK_REFERENCE(disparity_vx);

				//Compute disparity map
				NVXBM::NVXBMParams params;
				params.min_disparity = 0;
				params.max_disparity = 256;
				params.sad = 9;

				std::unique_ptr<NVXBM> stereo(NVXBM::createNVXBM(
					context, params, left_vx, right_vx, disparity_vx));

				stereo->run();

				vx_rectangle_t rect = { 0u, 0u, vx_uint32(size.width), vx_uint32(size.height) };
				NVXIO_SAFE_CALL(vxAccessImagePatch(disparity_vx, &rect, 0, &disparity_addr, (void**)&disparity.data, VX_READ_ONLY));
				NVXIO_SAFE_CALL(vxCommitImagePatch(disparity_vx, NULL, 0, &disparity_addr, (void**)&disparity.data));

				disparity = disparity.setTo(0, disparity <= 0);
				disparity.convertTo(disparity, CV_16UC1);
				disparity *= 256;

				disparities.push_back(disparity);
				times.push_back(std::round(stereo->getPerf()));


				std::cout << "NVXBM" << i << std::endl;
			}
		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/NVXBM/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());
		}
	}
#elif ALGO == GPUNVX_SGBM
	for (int x = 0; x < 2; x++)
	{ // NVXSGBM
		std::vector<cv::Mat> disparities;
		std::vector<float> times;

		for (int i = 0; i <= 193; i++){
			{
				nvxio::ContextGuard context;
				vxRegisterLogCallback(context, &myLogCallback, vx_false_e);

				cv::Size size = left_images[i].size();

				//Import left image to VX context 
				vx_imagepatch_addressing_t left_addr;
				left_addr.dim_x = vx_uint32(size.width);
				left_addr.dim_y = vx_uint32(size.height);
				left_addr.stride_x = vx_int32(sizeof(vx_uint8));
				left_addr.stride_y = vx_int32(left_images[i].step);

				void *left_ptrs[] = { left_images[i].data };

				vx_image left_vx = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &left_addr, left_ptrs, VX_IMPORT_TYPE_HOST);
				NVXIO_CHECK_REFERENCE(left_vx);

				//Import right image to VX context
				vx_imagepatch_addressing_t right_addr;
				right_addr.dim_x = vx_uint32(size.width);
				right_addr.dim_y = vx_uint32(size.height);
				right_addr.stride_x = vx_int32(sizeof(vx_uint8));
				right_addr.stride_y = vx_int32(right_images[i].step);

				void *right_ptrs[] = { right_images[i].data };

				vx_image right_vx = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &right_addr, right_ptrs, VX_IMPORT_TYPE_HOST);
				NVXIO_CHECK_REFERENCE(right_vx);

				//Create output image
				cv::Mat disparity = cv::Mat::zeros(size, CV_16UC1);

				vx_image disparity_vx = vxCreateImage(context, size.width, size.height, VX_DF_IMAGE_S16);
				vx_imagepatch_addressing_t disparity_addr;
				disparity_addr.dim_x = vx_uint32(size.width);
				disparity_addr.dim_y = vx_uint32(size.height);
				disparity_addr.stride_x = vx_int32(sizeof(vx_int16));
				disparity_addr.stride_y = vx_int32(disparity.step);

				NVXIO_CHECK_REFERENCE(disparity_vx);

				//Compute disparity map
				NVXSGBM::NVXSGBMParams params;
				params.max_disparity = 256;
				params.max_diff = 1;

				std::unique_ptr<NVXSGBM> stereo(NVXSGBM::createNVXSGBM(
					context, params, left_vx, right_vx, disparity_vx));

				stereo->run();

				vx_rectangle_t rect = { 0u, 0u, vx_uint32(size.width), vx_uint32(size.height) };
				NVXIO_SAFE_CALL(vxAccessImagePatch(disparity_vx, &rect, 0, &disparity_addr, (void**)&disparity.data, VX_READ_ONLY));
				NVXIO_SAFE_CALL(vxCommitImagePatch(disparity_vx, NULL, 0, &disparity_addr, (void**)&disparity.data));

				disparity = disparity.setTo(0, disparity <= 0);
				disparity.convertTo(disparity, CV_16UC1);
				disparity *= 256;

				disparities.push_back(disparity);
				times.push_back(std::round(stereo->getPerf()));


				std::cout << "NVXSGBM" << i << std::endl;

			}
		}

		for (int i = 0; i <= 193; i++){
			std::string file_name = "../kitteval/results/NVXSGBM/" + image_names[i];
			cv::imwrite(file_name, disparities[i]);
			std::string cmd = "exiftool.exe -Comment=" + float2str(times[i]) + " " + file_name;
			system(cmd.c_str());

		}
	}
#endif

	return 0;
}
