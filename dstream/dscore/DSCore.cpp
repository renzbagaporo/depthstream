#include "DSCore.h"
#include <opencv2\opencv.hpp>

DSCore::DSCore(){}

DSCore::~DSCore(){
	
	//Free textures
	cudaDestroyTextureObject(left_tex);
	cudaDestroyTextureObject(right_tex);
	cudaDestroyTextureObject(left_disp_tex);
	cudaDestroyTextureObject(right_disp_tex);

	//Free arrays
	cudaFreeArray(left_array);
	cudaFreeArray(right_array);
	cudaFreeArray(left_disp_array);
	cudaFreeArray(right_disp_array);

	//Free device memory
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(d_cost_vol_temp_a);
	cudaFree(d_cost_vol_temp_b);
	cudaFree(d_left_census);
	cudaFree(d_right_census);
	cudaFree(d_left_disp);
	cudaFree(d_right_disp);
	cudaFree(d_arm_vol);
	cudaFree(d_final_disp);
}

void DSCore::setup(int width, int height, int disparities){

	//Initialize variables
	this->width = width;
	this->height = height;

	if (disparities <= 64){
		this->disparities = 64;
	}
	else if (disparities > 64 && disparities <= 128){
		this->disparities = 128;
	}
	else if (disparities > 128 && disparities <= 256){
		this->disparities = 256;
	}
	else{
		this->disparities = 256;
	}

	//Allocate device memory
	cudaMalloc(&d_left, width * height * sizeof(unsigned char));
	cudaMalloc(&d_right, width * height * sizeof(unsigned char));
	cudaMalloc(&d_left_census, width * height * sizeof(unsigned long long int));
	cudaMalloc(&d_right_census, width * height * sizeof(unsigned long long int));
	cudaMalloc(&d_arm_vol, width * height * sizeof(uchar4));
	cudaMalloc(&d_cost_vol_temp_a, width * height * sizeof(float) * (disparities));
	cudaMalloc(&d_cost_vol_temp_b, width * height * sizeof(float) * (disparities));
	cudaMalloc(&d_left_disp, width * height * sizeof(unsigned short));
	cudaMalloc(&d_right_disp, width * height * sizeof(unsigned short));
	cudaMalloc(&d_final_disp, width * height * sizeof(unsigned short));

	//Initialize textures
	cudaChannelFormatDesc left_array_channel_desc = cudaCreateChannelDesc<unsigned char>(); cudaMallocArray(&left_array, &left_array_channel_desc, width, height);
	cudaChannelFormatDesc right_array_channel_desc = cudaCreateChannelDesc<unsigned char>(); cudaMallocArray(&right_array, &right_array_channel_desc, width, height);
	cudaChannelFormatDesc left_disp_array_channel_desc = cudaCreateChannelDesc<unsigned short>(); cudaMallocArray(&left_disp_array, &left_disp_array_channel_desc, width, height);
	cudaChannelFormatDesc right_disp_array_channel_desc = cudaCreateChannelDesc<unsigned short>(); cudaMallocArray(&right_disp_array, &right_disp_array_channel_desc, width, height);
	cudaChannelFormatDesc final_disp_array_channel_desc = cudaCreateChannelDesc<unsigned short>(); cudaMallocArray(&final_disp_array, &final_disp_array_channel_desc, width, height);

	cudaResourceDesc left_array_resc; memset(&left_array_resc, 0, sizeof(left_array_resc));
	cudaResourceDesc right_array_resc; memset(&right_array_resc, 0, sizeof(right_array_resc));
	cudaResourceDesc left_disp_array_resc; memset(&left_disp_array_resc, 0, sizeof(left_disp_array_resc));
	cudaResourceDesc right_disp_array_resc; memset(&right_disp_array_resc, 0, sizeof(right_disp_array_resc));
	cudaResourceDesc final_disp_array_resc; memset(&final_disp_array_resc, 0, sizeof(final_disp_array_resc));

	left_array_resc.resType = cudaResourceTypeArray; left_array_resc.res.array.array = left_array;
	right_array_resc.resType = cudaResourceTypeArray; right_array_resc.res.array.array = right_array;
	left_disp_array_resc.resType = cudaResourceTypeArray; left_disp_array_resc.res.array.array = left_disp_array;
	right_disp_array_resc.resType = cudaResourceTypeArray; right_disp_array_resc.res.array.array = right_disp_array;
	final_disp_array_resc.resType = cudaResourceTypeArray; final_disp_array_resc.res.array.array = final_disp_array;

	cudaTextureDesc left_array_tex_desc; memset(&left_array_tex_desc, 0, sizeof(left_array_tex_desc));
	cudaTextureDesc right_array_tex_desc; memset(&right_array_tex_desc, 0, sizeof(right_array_tex_desc));
	cudaTextureDesc left_disp_array_tex_desc; memset(&left_disp_array_tex_desc, 0, sizeof(left_disp_array_tex_desc));
	cudaTextureDesc right_disp_array_tex_desc; memset(&right_disp_array_tex_desc, 0, sizeof(right_disp_array_tex_desc));
	cudaTextureDesc final_disp_array_tex_desc; memset(&final_disp_array_tex_desc, 0, sizeof(final_disp_array_tex_desc));


	left_array_tex_desc.addressMode[0] = cudaAddressModeBorder;
	left_array_tex_desc.addressMode[1] = cudaAddressModeBorder;
	left_array_tex_desc.filterMode = cudaFilterModePoint;
	left_array_tex_desc.readMode = cudaReadModeElementType;
	left_array_tex_desc.normalizedCoords = 0;

	right_array_tex_desc.addressMode[0] = cudaAddressModeBorder;
	right_array_tex_desc.addressMode[1] = cudaAddressModeBorder;
	right_array_tex_desc.filterMode = cudaFilterModePoint;
	right_array_tex_desc.readMode = cudaReadModeElementType;
	right_array_tex_desc.normalizedCoords = 0;

	left_disp_array_tex_desc.addressMode[0] = cudaAddressModeBorder;
	left_disp_array_tex_desc.addressMode[1] = cudaAddressModeBorder;
	left_disp_array_tex_desc.filterMode = cudaFilterModePoint;
	left_disp_array_tex_desc.readMode = cudaReadModeElementType;
	left_disp_array_tex_desc.normalizedCoords = 0;

	right_disp_array_tex_desc.addressMode[0] = cudaAddressModeBorder;
	right_disp_array_tex_desc.addressMode[1] = cudaAddressModeBorder;
	right_disp_array_tex_desc.filterMode = cudaFilterModePoint;
	right_disp_array_tex_desc.readMode = cudaReadModeElementType;
	right_disp_array_tex_desc.normalizedCoords = 0;

	final_disp_array_tex_desc.addressMode[0] = cudaAddressModeBorder;
	final_disp_array_tex_desc.addressMode[1] = cudaAddressModeBorder;
	final_disp_array_tex_desc.filterMode = cudaFilterModePoint;
	final_disp_array_tex_desc.readMode = cudaReadModeElementType;
	final_disp_array_tex_desc.normalizedCoords = 0;

	left_tex = 0; cudaCreateTextureObject(&left_tex, &left_array_resc, &left_array_tex_desc, NULL);
	right_tex = 0; cudaCreateTextureObject(&right_tex, &right_array_resc, &right_array_tex_desc, NULL);
	left_disp_tex = 0; cudaCreateTextureObject(&left_disp_tex, &left_disp_array_resc, &left_disp_array_tex_desc, NULL);
	right_disp_tex = 0; cudaCreateTextureObject(&right_disp_tex, &right_disp_array_resc, &right_disp_array_tex_desc, NULL);
	final_disp_tex = 0; cudaCreateTextureObject(&final_disp_tex, &final_disp_array_resc, &final_disp_array_tex_desc, NULL);

}

void DSCore::copy_from_host_to_device(void *data_container, core_data data){
	switch (data)
	{
	case DSCore::LEFT_DATA:
		cudaMemcpy(d_left, data_container, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpyToArrayAsync(left_array, 0, 0, d_left, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
		break;
	case DSCore::RIGHT_DATA:
		cudaMemcpy(d_right, data_container, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpyToArrayAsync(right_array, 0, 0, d_right, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
		break;
	case DSCore::LEFT_CENSUS_DATA:
		cudaMemcpy(d_left_census, data_container, width * height * sizeof(uint2), cudaMemcpyHostToDevice);
		break;
	case DSCore::RIGHT_CENSUS_DATA:
		cudaMemcpy(d_right_census, data_container, width * height * sizeof(uint2), cudaMemcpyHostToDevice);
		break;
	case DSCore::ARM_DATA:
		cudaMemcpy(d_arm_vol, data_container, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
		break;
	case DSCore::COSTA_DATA:
		cudaMemcpy(d_cost_vol_temp_a, data_container, width * height * disparities * sizeof(float), cudaMemcpyHostToDevice);
		break;
	case DSCore::COSTB_DATA:
		cudaMemcpy(d_cost_vol_temp_b, data_container, width * height * disparities * sizeof(float), cudaMemcpyHostToDevice);
		break;
	case DSCore::LEFT_DISP_DATA:
		cudaMemcpy(d_left_disp, data_container, width * height * sizeof(unsigned short), cudaMemcpyHostToDevice);
		break;
	case DSCore::RIGHT_DISP_DATA:
		cudaMemcpy(d_right_disp, data_container, width * height * sizeof(unsigned short), cudaMemcpyHostToDevice);
		break;
	case DSCore::FINAL_DISP_DATA:
		cudaMemcpy(d_final_disp, data_container, width * height * sizeof(unsigned short), cudaMemcpyHostToDevice);
		break;
	default:
		break;
	}
}

void DSCore::copy_from_device_to_host(void *data_container, core_data data){
	switch (data)
	{
	case DSCore::LEFT_DATA:
		cudaMemcpyFromArray(data_container, left_array, 0, 0, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		break;
	case DSCore::RIGHT_DATA:
		cudaMemcpyFromArray(data_container, right_array, 0, 0, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		break;
	case DSCore::LEFT_CENSUS_DATA:
		cudaMemcpy(data_container, d_left_census, width * height * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
		break;
	case DSCore::RIGHT_CENSUS_DATA:
		cudaMemcpy(data_container, d_right_census, width * height * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
		break;
	case DSCore::ARM_DATA:
		cudaMemcpy(data_container, d_arm_vol, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		break;
	case DSCore::COSTA_DATA:
		cudaMemcpy(data_container, d_cost_vol_temp_a, width * height * disparities * sizeof(float), cudaMemcpyDeviceToHost);
		break;
	case DSCore::COSTB_DATA:
		cudaMemcpy(data_container, d_cost_vol_temp_b, width * height * disparities * sizeof(float), cudaMemcpyDeviceToHost);
		break;
	case DSCore::LEFT_DISP_DATA:
		cudaMemcpy(data_container, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
		break;
	case DSCore::RIGHT_DISP_DATA:
		cudaMemcpy(data_container, d_right_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
		break;
	case DSCore::FINAL_DISP_DATA:
		cudaMemcpy(data_container, d_final_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToHost);
		break;
	default:
		break;
	}
}

void DSCore::stereo_match(int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, float ad_gamma, float census_gamma, int disparity_tolerance, int region_voting_iterations){

	//Perform census transform
	census_transform(left_tex, d_left_census, width, height, 0);
	census_transform(right_tex, d_right_census, width, height, 0);

	//Create right cross
	cross_construct(right_tex, d_arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height, 0);

	//Match right to left
	match(d_left, d_right, d_left_census, d_right_census, d_cost_vol_temp_a, d_cost_vol_temp_b, d_arm_vol, d_right_disp, ad_gamma, census_gamma, false, width, height, disparities, 0);
	cudaMemcpyToArrayAsync(right_disp_array, 0, 0, d_right_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Create left cross
	cross_construct(left_tex, d_arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height, 0);

	//Match right to left
	match(d_left, d_right, d_left_census, d_right_census, d_cost_vol_temp_a, d_cost_vol_temp_b, d_arm_vol, d_left_disp, ad_gamma, census_gamma, true, width, height, disparities, 0);
	cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Check the consistency
	check_consistency(left_disp_tex, right_disp_tex, d_left_disp, disparity_tolerance, width, height, 0);
	cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Region voting
	for (int voting_iter = 0; voting_iter < region_voting_iterations; voting_iter++){
		if (voting_iter % 2 == 0){
			horizontal_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
			vertical_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		}
		else{
			vertical_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
			horizontal_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		}
	}

	//Median Filter
	median_filter(d_left_disp, d_final_disp, width, height);
}

void DSCore::stereo_match(int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, float ad_gamma, float census_gamma, int disparity_tolerance, int region_voting_iterations, int width, int height){

	//Perform census transform
	census_transform(left_tex, d_left_census, width, height, 0);
	census_transform(right_tex, d_right_census, width, height, 0);

	//Create right cross
	cross_construct(right_tex, d_arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height, 0);

	//Match right to left
	match(d_left, d_right, d_left_census, d_right_census, d_cost_vol_temp_a, d_cost_vol_temp_b, d_arm_vol, d_right_disp, ad_gamma, census_gamma, false, width, height, disparities, 0);
	cudaMemcpyToArrayAsync(right_disp_array, 0, 0, d_right_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Create left cross
	cross_construct(left_tex, d_arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height, 0);

	//Match right to left
	match(d_left, d_right, d_left_census, d_right_census, d_cost_vol_temp_a, d_cost_vol_temp_b, d_arm_vol, d_left_disp, ad_gamma, census_gamma, true, width, height, disparities, 0);
	cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Check the consistency
	check_consistency(left_disp_tex, right_disp_tex, d_left_disp, disparity_tolerance, width, height, 0);
	cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

	//Region voting
	for (int voting_iter = 0; voting_iter < region_voting_iterations; voting_iter++){
		if (voting_iter % 2 == 0){
			horizontal_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
			vertical_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		}
		else{
			vertical_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
			horizontal_voting(left_disp_tex, d_arm_vol, d_left_disp, width, height, 0);
			cudaMemcpyToArrayAsync(left_disp_array, 0, 0, d_left_disp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
		}
	}

	//Median Filter
	median_filter(d_left_disp, d_final_disp, width, height);
}