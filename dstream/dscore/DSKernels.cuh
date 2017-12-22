#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define OUTLIER 0

void census_transform(cudaTextureObject_t input_im, unsigned long long int *output_census, int width, int height, cudaStream_t stream);

void cross_construct(cudaTextureObject_t input_im, uchar4 *arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height, cudaStream_t stream);

void match(unsigned char *left, unsigned char *right,
	unsigned long long int *left_census, unsigned long long int *right_census,
	float *cost_vol_temp_a, float *cost_vol_temp_b, uchar4 *arm_vol, unsigned short *disp_im, float ad_gamma,
	float census_gamma, bool left_to_right, int width, int height, int max_disparity, cudaStream_t stream);

void check_consistency(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned short *output_disp_im, int disparity_tolerance, int width, int height, cudaStream_t stream);

void horizontal_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height, cudaStream_t stream);

void vertical_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height, cudaStream_t stream);

void median_filter(unsigned short *input_disp, unsigned short *output_disp, int width, int height);
