#pragma once
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DSKernels.cuh"

class DSCore{
private:
	//Stereo parameters
	int width, height, disparities;

	//Device vars
	unsigned char *d_left;
	unsigned char *d_right;
	unsigned long long int *d_left_census;
	unsigned long long int *d_right_census;
	uchar4 *d_arm_vol;
	float *d_cost_vol_temp_a;
	float *d_cost_vol_temp_b;
	unsigned short *d_left_disp;
	unsigned short *d_right_disp;
	unsigned short *d_final_disp;

	//CUDA arrays
	cudaArray *left_array;
	cudaArray *right_array;
	cudaArray *left_disp_array;
	cudaArray *right_disp_array;
	cudaArray *final_disp_array;

	//Texture objects
	cudaTextureObject_t left_tex;
	cudaTextureObject_t right_tex;
	cudaTextureObject_t left_disp_tex;
	cudaTextureObject_t right_disp_tex;
	cudaTextureObject_t final_disp_tex;

public:
	DSCore();
	~DSCore();

	void setup(int width, int height, int disparities);

	//Data available
	enum core_data{ LEFT_DATA, RIGHT_DATA, LEFT_CENSUS_DATA, RIGHT_CENSUS_DATA, ARM_DATA, COSTA_DATA, COSTB_DATA, LEFT_DISP_DATA, RIGHT_DISP_DATA, FINAL_DISP_DATA};

	//Synchronous copy methods
	void copy_from_device_to_host(void *data_container, core_data data);
	void copy_from_host_to_device(void *data_container, core_data data);

	//Class methods
	void stereo_match(int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, float ad_gamma, float census_gamma, int disparity_tolerance, int region_voting_iterations);
	void stereo_match(int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, float ad_gamma, float census_gamma, int disparity_tolerance, int region_voting_iterations, int width, int height);
};