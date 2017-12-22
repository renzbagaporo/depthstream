#pragma once
#include "kernels.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GRANULARITY 4
#define OUTLIER 0

void census_transform(cudaTextureObject_t input_im, uint2 *output_census, int width, int height, cudaStream_t stream);
void cross_construct(cudaTextureObject_t input_im, uchar4 *arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height, cudaStream_t stream);
void cross_construct_bgr(cudaTextureObject_t input_im, uchar4 *arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height, cudaStream_t stream);
void match(cudaTextureObject_t left_tex, cudaTextureObject_t left_census_tex, cudaTextureObject_t right_tex, cudaTextureObject_t right_census_tex, float4 *cost_vol_temp_a, float4 *cost_vol_temp_b, uchar4 *arm_vol, unsigned char *disp_im, float ad_gamma, float census_gamma, bool left_to_right, int width, int height, int max_disparity, cudaStream_t stream);
void check_consistency(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned char *output_disp_im, int disparity_tolerance, int width, int height, cudaStream_t stream);
void horizontal_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height, cudaStream_t stream);
void vertical_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height, cudaStream_t stream);
void extrapolation(cudaTextureObject_t input_disp, unsigned char *output_disp, int width, int height, cudaStream_t stream);
void color2gray(cudaTextureObject_t input, unsigned char *output, int width, int height, cudaStream_t stream);
void cleanup(unsigned char *input, unsigned char *output, int width, int height, int min_disparity, cudaStream_t stream);