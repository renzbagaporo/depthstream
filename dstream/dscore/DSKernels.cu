#include "DSKernels.cuh"
#include <opencv2\opencv.hpp>

//#define KERN_DEB

#define BLOCK_X  16
#define BLOCK_Y  16

// Exchange trick: Morgan McGuire, ShaderX 2008
#define s2(a,b)            { unsigned short tmp = a; a = min(a,b); b = max(tmp,b); }
#define mn3(a,b,c)         s2(a,b); s2(a,c);
#define mx3(a,b,c)         s2(b,c); s2(a,c);

#define mnmx3(a,b,c)       mx3(a,b,c); s2(a,b);                               // 3 exchanges
#define mnmx4(a,b,c,d)     s2(a,b); s2(c,d); s2(a,c); s2(b,d);                // 4 exchanges
#define mnmx5(a,b,c,d,e)   s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e);          // 6 exchanges
#define mnmx6(a,b,c,d,e,f) s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); // 7 exchanges

#define SMEM(x,y)  smem[(x)+1][(y)+1]
#define IN(x,y)    d_in[(y)*nx + (x)]

/////////////////////////////////////////////////////////////////////////////Helpers/////////////////////////////////////////////////////////////////////////////

#ifdef KERN_DEB
#include <iostream>
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
#endif

#define DIVIDE_UP(a, b) (int)std::ceil((float)a / (float)b)

/////////////////////////////////////////////////////////////////////////////Kernels/////////////////////////////////////////////////////////////////////////////

__global__
void census_transform_kernel(cudaTextureObject_t input_im, unsigned long long int *output_census, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){
		unsigned char ref = tex2D<unsigned char>(input_im, image_col, image_row);

		unsigned int sum1 = 0x0000;
		sum1 =
			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 4) > ref) << 31) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 4) > ref) << 30) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 4) > ref) << 29) |
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 4) > ref) << 28) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 4) > ref) << 27) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 4) > ref) << 26) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 4) > ref) << 25) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 3) > ref) << 24) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 3) > ref) << 23) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 3) > ref) << 22) |
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 3) > ref) << 21) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 3) > ref) << 20) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 3) > ref) << 19) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 3) > ref) << 18) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 2) > ref) << 17) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 2) > ref) << 16) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 2) > ref) << 15) |
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 2) > ref) << 14) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 2) > ref) << 13) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 2) > ref) << 12) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 2) > ref) << 11) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 1) > ref) << 10) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 1) > ref) << 9) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 1) > ref) << 8) |
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 1) > ref) << 7) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 1) > ref) << 6) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 1) > ref) << 5) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 1) > ref) << 4) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 0) > ref) << 3) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 0) > ref) << 2) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 0) > ref) << 1) |
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 0) > ref) << 0);

		unsigned int sum2 = 0x0000;
		sum2 =
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 0) > ref) << 31) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 0) > ref) << 30) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 0) > ref) << 29) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 0) > ref) << 28) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 1) > ref) << 27) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 1) > ref) << 26) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 1) > ref) << 25) |
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 1) > ref) << 24) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 1) > ref) << 23) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 1) > ref) << 22) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 1) > ref) << 21) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 2) > ref) << 20) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 2) > ref) << 19) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 2) > ref) << 18) |
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 2) > ref) << 17) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 2) > ref) << 16) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 2) > ref) << 15) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 2) > ref) << 14) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 3) > ref) << 13) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 3) > ref) << 12) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 3) > ref) << 11) |
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 3) > ref) << 10) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 3) > ref) << 9) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 3) > ref) << 8) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 3) > ref) << 7) |

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 4) > ref) << 6) |
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 4) > ref) << 5) |
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 4) > ref) << 4) |
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 4) > ref) << 3) |
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 4) > ref) << 2) |
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 4) > ref) << 1) |
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 4) > ref) << 0);

		uint2 temp = make_uint2(sum1, sum2);
		output_census[image_row * width + image_col] = *reinterpret_cast<unsigned long long int*>(&temp);
	}
}

__global__
void cross_construct_kernel(cudaTextureObject_t input_im, uchar4 *ouput_arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height){
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;

	if (image_row < height && image_col < width){

		uchar4 pix_arm = make_uchar4(0, 0, 0, 0);

		int ref = tex2D<unsigned char>(input_im, image_col, image_row);
		int scan_length, diff_curr_ref, diff_curr_next;

		//Upward scan
		scan_length = 0; diff_curr_ref = 0; diff_curr_next = 0;
		while (true)
		{
			diff_curr_ref = abs(ref - tex2D<unsigned char>(input_im, image_col, image_row - scan_length));
			diff_curr_next = abs(ref - tex2D<unsigned char>(input_im, image_col, image_row - scan_length - 1));

			if (!(scan_length < max_arm_length &&
				image_row - scan_length > 0 &&
				diff_curr_ref <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold) &&
				diff_curr_next <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold))) break;

			scan_length++;
		}

		pix_arm.x = scan_length;

		//Downward scan
		scan_length = 0; diff_curr_ref = 0; diff_curr_next = 0;
		while (true)
		{
			diff_curr_ref = abs(ref - tex2D<unsigned char>(input_im, image_col, image_row + scan_length));
			diff_curr_next = abs(ref - tex2D<unsigned char>(input_im, image_col, image_row + scan_length + 1));

			if (!(scan_length < max_arm_length &&
				image_row + scan_length < height - 1 &&
				diff_curr_ref <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold) &&
				diff_curr_next <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold))) break;

			scan_length++;
		}

		pix_arm.y = scan_length;

		//Leftward scan
		scan_length = 0; diff_curr_ref = 0; diff_curr_next = 0;
		while (true)
		{
			diff_curr_ref = abs(ref - tex2D<unsigned char>(input_im, image_col - scan_length, image_row));
			diff_curr_next = abs(ref - tex2D<unsigned char>(input_im, image_col - scan_length - 1, image_row));

			if (!(scan_length < max_arm_length &&
				image_col - scan_length > 0 &&
				diff_curr_ref <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold) &&
				diff_curr_next <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold))) break;

			scan_length++;
		}

		pix_arm.z = scan_length;

		//Rightward scan
		scan_length = 0; diff_curr_ref = 0; diff_curr_next = 0;
		while (true)
		{
			diff_curr_ref = abs(ref - tex2D<unsigned char>(input_im, image_col + scan_length, image_row));
			diff_curr_next = abs(ref - tex2D<unsigned char>(input_im, image_col + scan_length + 1, image_row));

			if (!(scan_length < max_arm_length &&
				image_col + scan_length < width &&
				diff_curr_ref <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold) &&
				diff_curr_next <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold))) break;

			scan_length++;
		}
		pix_arm.w = scan_length;

		pix_arm.x = pix_arm.x == 0 ? (image_row - 2 >= 0 ? 2 : 0) : pix_arm.x;
		pix_arm.y = pix_arm.y == 0 ? (image_row + 2 < height ? 2 : 0) : pix_arm.y;

		//pix_arm.x = image_row - 2 >= 0 ? 2 : pix_arm.x;
		//pix_arm.y = image_row + 2 < height ? 2 : pix_arm.y;

		pix_arm.z = pix_arm.z == 0 ? (image_col - 2 >= 0 ? 2 : 0) : pix_arm.z;
		pix_arm.w = pix_arm.w == 0 ? (image_col + 2 < width ? 2 : 0) : pix_arm.w;

		ouput_arm_vol[image_row * width + image_col] = pix_arm;
	}
}

__global__
void cost_initialization_kernel(unsigned char *left, unsigned char *right, unsigned long long int *left_census, unsigned long long int *right_census, float *cost_vol, float ad_gamma, float census_gamma, bool left_to_right, int width, int height){
	extern __shared__ unsigned char temp[];

	unsigned char *ref_temp = temp;
	unsigned char *targ_temp = &ref_temp[blockDim.x];

	unsigned long long int *ref_census_temp = (unsigned long long int*)&targ_temp[blockDim.x * 2];
	unsigned long long int *targ_census_temp = &ref_census_temp[blockDim.x];

	//Initialize to zero
	ref_temp[threadIdx.x] = 0;
	targ_temp[threadIdx.x] = 0;
	targ_temp[blockDim.x + threadIdx.x] = 0;

	ref_census_temp[threadIdx.x] = 0;
	targ_census_temp[threadIdx.x] = 0;
	targ_census_temp[blockDim.x + threadIdx.x] = 0;

	__syncthreads();

	int image_row = blockIdx.y;

	float cost = 0.0f;

	if (image_row < height){

		if (left_to_right){
			for (int image_col = 0; image_col < width; image_col++){

				int block_index = image_col % blockDim.x;

				if (block_index == 0){
					if (image_col + threadIdx.x < width){
						ref_temp[threadIdx.x] = left[image_row * width + image_col + threadIdx.x];
						ref_census_temp[threadIdx.x] = left_census[image_row * width + image_col + threadIdx.x];
					}

					if (image_col + threadIdx.x < width){
						targ_temp[blockDim.x + threadIdx.x] = right[image_row * width + image_col + threadIdx.x];
						targ_census_temp[blockDim.x + threadIdx.x] = right_census[image_row * width + image_col + threadIdx.x];
					}

					if ((int)(image_col - blockDim.x + threadIdx.x) >= 0 && (int)(image_col - blockDim.x + threadIdx.x) < width){
						targ_temp[threadIdx.x] = right[image_row * width + image_col - blockDim.x + threadIdx.x];
						targ_census_temp[threadIdx.x] = right_census[image_row * width + image_col - blockDim.x + threadIdx.x];
					}
					__syncthreads();
				}

				float ad_cost, census_cost;

				ad_cost = (fabsf(ref_temp[block_index] - targ_temp[blockDim.x + block_index - threadIdx.x]) / 255.0f) * ad_gamma;
				census_cost = (__popcll(ref_census_temp[block_index] ^ targ_census_temp[blockDim.x + block_index - threadIdx.x]) / 64.0f) * census_gamma;

				cost += ad_cost + census_cost;

				cost_vol[image_row * width * blockDim.x + image_col * blockDim.x + threadIdx.x] = cost;
			}
		}
		else{

			for (int image_col = 0; image_col < width; image_col++){

				int block_index = image_col % blockDim.x;

				if (block_index == 0){

					if (image_col + threadIdx.x < width){
						ref_temp[threadIdx.x] = right[image_row * width + image_col + threadIdx.x];
						ref_census_temp[threadIdx.x] = right_census[image_row * width + image_col + threadIdx.x];
					}

					if (image_col + threadIdx.x < width){
						targ_temp[threadIdx.x] = left[image_row * width + image_col + threadIdx.x];
						targ_census_temp[threadIdx.x] = left_census[image_row * width + image_col + threadIdx.x];
					}

					if (image_col + blockDim.x + threadIdx.x < width){
						targ_temp[blockDim.x + threadIdx.x] = left[image_row * width + image_col + blockDim.x + threadIdx.x];
						targ_census_temp[blockDim.x + threadIdx.x] = left_census[image_row * width + image_col + blockDim.x + threadIdx.x];
					}

					__syncthreads();
				}

				float ad_cost, census_cost;

				ad_cost = (fabsf(ref_temp[block_index] - targ_temp[block_index + threadIdx.x]) / 255.0f) * ad_gamma;
				census_cost = (__popcll(ref_census_temp[block_index] ^ targ_census_temp[block_index + threadIdx.x]) / 64.0f) * census_gamma;

				cost += ad_cost + census_cost;

				cost_vol[image_row * width * blockDim.x + image_col * blockDim.x + threadIdx.x] = cost;
			}
		}
	}
}

__global__
void horizontal_aggregation_kernel(float *cost_vol_in, uchar4 *arm_vol, float *cost_vol_out, int width, int height){

	int image_col = blockIdx.x;

	float sum = 0.0f;

	for (int image_row = 0; image_row < height; image_row++){

		uchar4 pixel_arm = arm_vol[image_row * width + image_col];

		int right_limit = image_col + pixel_arm.w;
		int left_limit = image_col - pixel_arm.z - 1;

		float aggregate = cost_vol_in[image_row * width * blockDim.x + right_limit * blockDim.x + threadIdx.x];

		if (left_limit >= 0)
			aggregate -= cost_vol_in[image_row * width * blockDim.x + left_limit * blockDim.x + threadIdx.x];

		sum += aggregate;

		cost_vol_out[image_row * width * blockDim.x + image_col * blockDim.x + threadIdx.x] = sum;
	}
}

__global__
void vertical_aggregation_kernel(float *cost_vol_in, uchar4 *arm_vol, float *cost_vol_out, unsigned short *disp_im, int width, int height){

	int image_row = blockIdx.y;

	__shared__ unsigned int reduce_cache[32];
	__shared__ float cost_cache[256];

	

	for (int image_col = 0; image_col < width; image_col++){

		uchar4 pix_arm = arm_vol[image_row * width + image_col];

		int down_lim = image_row + pix_arm.y;
		int up_lim = image_row - pix_arm.x - 1;

		float aggregate = cost_vol_in[down_lim * width * blockDim.x + image_col * blockDim.x + threadIdx.x];

		if (up_lim >= 0)
			aggregate -= cost_vol_in[up_lim * width * blockDim.x + image_col * blockDim.x + threadIdx.x];

		//cost_vol_out[image_row * width * blockDim.x + image_col * blockDim.x + threadIdx.x] = aggregate;

		cost_cache[threadIdx.x] = aggregate;
		//Find the minimum

		unsigned int min_cost = (((unsigned int)(aggregate * 10000)) << 8) | threadIdx.x;
		unsigned int temp_min_cost = 0;
		

		int lane = threadIdx.x % 32;
		int wid = threadIdx.x / 32;

		temp_min_cost = __shfl_down(min_cost, 16);
		min_cost = min(min_cost, temp_min_cost);

		temp_min_cost = __shfl_down(min_cost, 8);
		min_cost = min(min_cost, temp_min_cost);

		temp_min_cost = __shfl_down(min_cost, 4);
		min_cost = min(min_cost, temp_min_cost);

		temp_min_cost = __shfl_down(min_cost, 2);
		min_cost = min(min_cost, temp_min_cost);

		temp_min_cost = __shfl_down(min_cost, 1);
		min_cost = min(min_cost, temp_min_cost);

		if (lane == 0) reduce_cache[wid] = min_cost;

		__syncthreads();

		min_cost = (threadIdx.x < blockDim.x / 32) ? reduce_cache[lane] : UINT_MAX;

		if (wid == 0){

			temp_min_cost = __shfl_down(min_cost, 4);
			min_cost = min(min_cost, temp_min_cost);

			temp_min_cost = __shfl_down(min_cost, 2);
			min_cost = min(min_cost, temp_min_cost);

			temp_min_cost = __shfl_down(min_cost, 1);
			min_cost = min(min_cost, temp_min_cost);

		}

		if (threadIdx.x == 0){
			unsigned short disp = (unsigned short)((min_cost & 0x000000FF));
			if (disp >= 1 && disp < blockDim.x - 1)
				disp_im[image_row * width + image_col] = (unsigned short)((disp + ((cost_cache[disp + 1] - cost_cache[disp - 1]) / (2 * (-cost_cache[disp + 1] - cost_cache[disp - 1] + 2 * cost_cache[disp])))) * 256.0f);
			else
				disp_im[image_row * width + image_col] = disp << 8;
		}
	}
}

__global__
void consistency_check_kernel(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned short *output_disp_im, int disparity_tolerance, int width, int height){

	int col_to_access = blockIdx.x * blockDim.x + threadIdx.x;
	int row_to_access = blockIdx.y * blockDim.y + threadIdx.y;

	if (row_to_access < height && col_to_access < width){
		unsigned short disp = tex2D<unsigned short>(left_disp_im, col_to_access, row_to_access);
		unsigned short to_check = tex2D<unsigned short>(right_disp_im, col_to_access - (disp >> 8), row_to_access);

		output_disp_im[row_to_access * width + col_to_access] = (abs(disp - to_check) <= disparity_tolerance * 256) ? disp : OUTLIER;
	}
}

__global__
void horizontal_voting_kernel(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){

		int sums[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		int eligible_votes = 0;
		int no_of_votes = 0;

		//Load arm data
		uchar4 pix_arm = arm_vol[image_col + image_row * width];

		//Check disp value
		int disp_value = tex2D<unsigned short>(input_disp, image_col, image_row);

		if (disp_value == OUTLIER){
			for (int pix_iter = -pix_arm.z; pix_iter <= pix_arm.w; pix_iter++){
				int disp_val = tex2D<unsigned short>(input_disp, image_col + pix_iter, image_row);
				if (disp_val != OUTLIER){
					sums[0] += ((disp_val & 1) != 0);
					sums[1] += ((disp_val & 2) != 0);
					sums[2] += ((disp_val & 4) != 0);
					sums[3] += ((disp_val & 8) != 0);
					sums[4] += ((disp_val & 16) != 0);
					sums[5] += ((disp_val & 32) != 0);
					sums[6] += ((disp_val & 64) != 0);
					sums[7] += ((disp_val & 128) != 0);
					sums[8] += ((disp_val & 256) != 0);
					sums[9] += ((disp_val & 512) != 0);
					sums[10] += ((disp_val & 1024) != 0);
					sums[11] += ((disp_val & 2048) != 0);
					sums[12] += ((disp_val & 4096) != 0);
					sums[13] += ((disp_val & 8192) != 0);
					sums[14] += ((disp_val & 16384) != 0);
					sums[15] += ((disp_val & 32768) != 0);
					eligible_votes++;
				}
				no_of_votes++;
			}
			__syncthreads();

			int majority = eligible_votes * 0.5;
			disp_value = (
				((sums[15] > majority) << 15) +
				((sums[14] > majority) << 14) +
				((sums[13] > majority) << 13) +
				((sums[12] > majority) << 12) +
				((sums[11] > majority) << 11) +
				((sums[10] > majority) << 10) +
				((sums[9] > majority) << 9) +
				((sums[8] > majority) << 8) +
				((sums[7] > majority) << 7) +
				((sums[6] > majority) << 6) +
				((sums[5] > majority) << 5) +
				((sums[4] > majority) << 4) +
				((sums[3] > majority) << 3) +
				((sums[2] > majority) << 2) +
				((sums[1] > majority) << 1) +
				((sums[0] > majority) << 0));
			disp_value = (eligible_votes > no_of_votes * 0.35f) ? disp_value : OUTLIER;
		}

		output_disp[image_col + image_row * width] = disp_value;
	}
}

__global__
void vertical_voting_kernel(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){

		int sums[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		int eligible_votes = 0;
		int no_of_votes = 0;

		//Load arm data
		uchar4 pix_arm = arm_vol[image_col + image_row * width];

		//Check disp value
		int disp_value = tex2D<unsigned short>(input_disp, image_col, image_row);

		if (disp_value == OUTLIER){
			for (int pix_iter = -pix_arm.x; pix_iter <= pix_arm.y; pix_iter++){
				int disp_val = tex2D<unsigned char>(input_disp, image_col, image_row + pix_iter);
				if (disp_val != OUTLIER){
					sums[0] += ((disp_val & 1) != 0);
					sums[1] += ((disp_val & 2) != 0);
					sums[2] += ((disp_val & 4) != 0);
					sums[3] += ((disp_val & 8) != 0);
					sums[4] += ((disp_val & 16) != 0);
					sums[5] += ((disp_val & 32) != 0);
					sums[6] += ((disp_val & 64) != 0);
					sums[7] += ((disp_val & 128) != 0);
					sums[8] += ((disp_val & 256) != 0);
					sums[9] += ((disp_val & 512) != 0);
					sums[10] += ((disp_val & 1024) != 0);
					sums[11] += ((disp_val & 2048) != 0);
					sums[12] += ((disp_val & 4096) != 0);
					sums[13] += ((disp_val & 8192) != 0);
					sums[14] += ((disp_val & 16384) != 0);
					sums[15] += ((disp_val & 32768) != 0);
					eligible_votes++;
				}
				no_of_votes++;
			}
			__syncthreads();

			int majority = eligible_votes * 0.5;
			disp_value = (
				((sums[15] > majority) << 15) +
				((sums[14] > majority) << 14) +
				((sums[13] > majority) << 13) +
				((sums[12] > majority) << 12) +
				((sums[11] > majority) << 11) +
				((sums[10] > majority) << 10) +
				((sums[9] > majority) << 9) +
				((sums[8] > majority) << 8) +
				((sums[7] > majority) << 7) +
				((sums[6] > majority) << 6) +
				((sums[5] > majority) << 5) +
				((sums[4] > majority) << 4) +
				((sums[3] > majority) << 3) +
				((sums[2] > majority) << 2) +
				((sums[1] > majority) << 1) +
				((sums[0] > majority) << 0));
			disp_value = (eligible_votes > no_of_votes * 0.35f) ? disp_value : OUTLIER;
		}

		output_disp[image_col + image_row * width] = disp_value;
	}
}


__global__
void median_filter_kernel(unsigned short *d_in, unsigned short *d_out, int nx, int ny)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	// guards: is at boundary?
	bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCK_X - 1);
	bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCK_Y - 1);

	__shared__ unsigned short smem[BLOCK_X + 2][BLOCK_Y + 2];
	// clear out shared memory (zero padding)
	if (is_x_top)           SMEM(tx - 1, ty) = 0;
	else if (is_x_bot)      SMEM(tx + 1, ty) = 0;
	if (is_y_top) {
		SMEM(tx, ty - 1) = 0;
		if (is_x_top)       SMEM(tx - 1, ty - 1) = 0;
		else if (is_x_bot)  SMEM(tx + 1, ty - 1) = 0;
	}
	else if (is_y_bot) {
		SMEM(tx, ty + 1) = 0;
		if (is_x_top)       SMEM(tx - 1, ty + 1) = 0;
		else if (is_x_bot)  SMEM(tx + 1, ty + 1) = 0;
	}

	// guards: is at boundary and still more image?
	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;
	is_x_top &= (x > 0); is_x_bot &= (x < nx - 1);
	is_y_top &= (y > 0); is_y_bot &= (y < ny - 1);

	// each thread pulls from image
	SMEM(tx, ty) = IN(x, y); // self
	if (is_x_top)           SMEM(tx - 1, ty) = IN(x - 1, y);
	else if (is_x_bot)      SMEM(tx + 1, ty) = IN(x + 1, y);
	if (is_y_top) {
		SMEM(tx, ty - 1) = IN(x, y - 1);
		if (is_x_top)       SMEM(tx - 1, ty - 1) = IN(x - 1, y - 1);
		else if (is_x_bot)  SMEM(tx + 1, ty - 1) = IN(x + 1, y - 1);
	}
	else if (is_y_bot) {
		SMEM(tx, ty + 1) = IN(x, y + 1);
		if (is_x_top)       SMEM(tx - 1, ty + 1) = IN(x - 1, y + 1);
		else if (is_x_bot)  SMEM(tx + 1, ty + 1) = IN(x + 1, y + 1);
	}
	__syncthreads();

	// pull top six from shared memory
	unsigned short v[6] = { SMEM(tx - 1, ty - 1), SMEM(tx, ty - 1), SMEM(tx + 1, ty - 1),
		SMEM(tx - 1, ty), SMEM(tx, ty), SMEM(tx + 1, ty) };

	// with each pass, remove min and max values and add new value
	mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
	v[5] = SMEM(tx - 1, ty + 1); // add new contestant
	mnmx5(v[1], v[2], v[3], v[4], v[5]);
	v[5] = SMEM(tx, ty + 1);
	mnmx4(v[2], v[3], v[4], v[5]);
	v[5] = SMEM(tx + 1, ty + 1);
	mnmx3(v[3], v[4], v[5]);

	// pick the middle one
	d_out[y*nx + x] = v[4];
}


/////////////////////////////////////////////////////////////////////////////Stubs/////////////////////////////////////////////////////////////////////////////

void census_transform(cudaTextureObject_t input_im, unsigned long long int *output_census, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	census_transform_kernel << <blocks, threads >> >(input_im, output_census, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Census transform failed.");
#endif
}

void cross_construct(cudaTextureObject_t input_im, uchar4 *arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	cross_construct_kernel << <blocks, threads >> >(input_im, arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Cross construct failed.");
#endif
}

void match(unsigned char *left, unsigned char *right,
	unsigned long long int *left_census, unsigned long long int *right_census, float *cost_vol_temp_a, float *cost_vol_temp_b, uchar4 *arm_vol,
	unsigned short *disp_im, float gamma, float census_gamma, bool left_to_right, int width, int height, int max_disparity, cudaStream_t stream){

	dim3 b(1, height); dim3 t(max_disparity);
	size_t mem_sz = t.x * (sizeof(unsigned long long int) + sizeof(unsigned char)) * 3;
	cost_initialization_kernel << <b, t, mem_sz >> >(left, right, left_census, right_census, (float*)cost_vol_temp_a, gamma, census_gamma, left_to_right, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Cost initialization failed.");
#endif

	b = dim3(width);
	t = dim3(max_disparity);
	horizontal_aggregation_kernel << < b, t >> > ((float*)cost_vol_temp_a, arm_vol, (float*)cost_vol_temp_b, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal aggregation failed.");
#endif

	b = dim3(1, height);
	t = dim3(max_disparity);
	vertical_aggregation_kernel << < b, t >> > ((float*)cost_vol_temp_b, arm_vol, (float*)cost_vol_temp_a, disp_im, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal aggregation failed.");
#endif
}

void check_consistency(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned short *output_disp_im, int disparity_tolerance, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	consistency_check_kernel << <blocks, threads >> >(left_disp_im, right_disp_im, output_disp_im, disparity_tolerance, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Consistency check failed.");
#endif
}

void horizontal_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	horizontal_voting_kernel << <blocks, threads >> >(input_disp, arm_vol, output_disp, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal voting failed.");
#endif
}

void vertical_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned short *output_disp, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	vertical_voting_kernel << <blocks, threads >> >(input_disp, arm_vol, output_disp, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Vertical voting failed.");
#endif
}

void median_filter(unsigned short *input_disp, unsigned short *output_disp, int width, int height){
	dim3 blocks(width / BLOCK_X, height / BLOCK_Y);
	dim3 threads(BLOCK_X, BLOCK_Y);
	median_filter_kernel << <blocks, threads >> >(input_disp, output_disp, width, height);
}


