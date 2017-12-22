#include "kernels.cuh"

//#define KERN_DEB

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
__forceinline__ __device__ unsigned long long int int2_to_ll(uint2 i){ return  __double_as_longlong(__hiloint2double(i.x, i.y)); }
__forceinline__ __device__ unsigned int uchar3_max_diff(uchar3 a, uchar3 b){ return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z))); }
__forceinline__ __device__ float4 add_float4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__forceinline__ __device__ float4 subtract_float4(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

/////////////////////////////////////////////////////////////////////////////Kernels/////////////////////////////////////////////////////////////////////////////

__global__
void color2gray_kernel(cudaTextureObject_t input, unsigned char *output, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	int channels = 3;

	if (image_col < width && image_row < height){

		int b_pix = tex2D<unsigned char>(input, image_col * channels + 0, image_row);
		int g_pix = tex2D<unsigned char>(input, image_col * channels + 1, image_row);
		int r_pix = tex2D<unsigned char>(input, image_col * channels + 2, image_row);

		int gray = (int)((b_pix + g_pix + r_pix) / 3.0f);
		output[image_col + image_row * width] = (unsigned char)gray;
	}
}

__global__
void census_transform_kernel(cudaTextureObject_t input_im, uint2 *output_census, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){
		unsigned char ref = tex2D<unsigned char>(input_im, image_col, image_row);
		/*unsigned int sum1 = 0, sum2 = 0;
		
		sum1 =
			((tex2D<unsigned char>(input_im, image_col, image_row - 2) > ref) << 31) +
			((tex2D<unsigned char>(input_im, image_col, image_row - 1) > ref) << 30) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row) > ref) << 29) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row) > ref) << 28) +
			((tex2D<unsigned char>(input_im, image_col, image_row + 1) > ref) << 27) +
			((tex2D<unsigned char>(input_im, image_col, image_row + 2) > ref) << 26);

			*/
		unsigned int sum1 = 0x0000;
		sum1 =
			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 4) > ref) << 31) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 4) > ref) << 30) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 4) > ref) << 29) +
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 4) > ref) << 28) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 4) > ref) << 27) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 4) > ref) << 26) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 4) > ref) << 25) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 3) > ref) << 24) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 3) > ref) << 23) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 3) > ref) << 22) +
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 3) > ref) << 21) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 3) > ref) << 20) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 3) > ref) << 19) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 3) > ref) << 18) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 2) > ref) << 17) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 2) > ref) << 16) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 2) > ref) << 15) +
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 2) > ref) << 14) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 2) > ref) << 13) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 2) > ref) << 12) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 2) > ref) << 11) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 1) > ref) << 10) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 1) > ref) << 9) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 1) > ref) << 8) +
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 1) > ref) << 7) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row - 1) > ref) << 6) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row - 1) > ref) << 5) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row - 1) > ref) << 4) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row - 0) > ref) << 3) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row - 0) > ref) << 2) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row - 0) > ref) << 1) +
			((tex2D<unsigned char>(input_im, image_col - 0, image_row - 0) > ref) << 0);

		unsigned int sum2 = 0x0000;
		sum2 =
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 0) > ref) << 31) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 0) > ref) << 30) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 0) > ref) << 29) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 0) > ref) << 28) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 1) > ref) << 27) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 1) > ref) << 26) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 1) > ref) << 25) +
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 1) > ref) << 24) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 1) > ref) << 23) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 1) > ref) << 22) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 1) > ref) << 21) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 2) > ref) << 20) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 2) > ref) << 19) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 2) > ref) << 18) +
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 2) > ref) << 17) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 2) > ref) << 16) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 2) > ref) << 15) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 2) > ref) << 14) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 3) > ref) << 13) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 3) > ref) << 12) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 3) > ref) << 11) +
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 3) > ref) << 10) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 3) > ref) << 9) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 3) > ref) << 8) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 3) > ref) << 7) +

			((tex2D<unsigned char>(input_im, image_col - 3, image_row + 4) > ref) << 6) +
			((tex2D<unsigned char>(input_im, image_col - 2, image_row + 4) > ref) << 5) +
			((tex2D<unsigned char>(input_im, image_col - 1, image_row + 4) > ref) << 4) +
			((tex2D<unsigned char>(input_im, image_col + 0, image_row + 4) > ref) << 3) +
			((tex2D<unsigned char>(input_im, image_col + 1, image_row + 4) > ref) << 2) +
			((tex2D<unsigned char>(input_im, image_col + 2, image_row + 4) > ref) << 1) +
			((tex2D<unsigned char>(input_im, image_col + 3, image_row + 4) > ref) << 0);

		output_census[image_row * width + image_col] = make_uint2(sum1, sum2);
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
		pix_arm.z = pix_arm.z == 0 ? (image_col - 2 >= 0 ? 2 : 0) : pix_arm.z;
		pix_arm.w = pix_arm.w == 0 ? (image_col + 2 < width ? 2 : 0) : pix_arm.w;

		ouput_arm_vol[image_row * width + image_col] = pix_arm;
	}
}

__global__
void cross_construct_kernel_bgr(cudaTextureObject_t input_im, uchar4 *ouput_arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height){
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;

	int channels = 3;

	if (image_row < height && image_col < width){

		uchar4 pix_arm = make_uchar4(0, 0, 0, 0);

		uchar3 ref = make_uchar3(
			tex2D<unsigned char>(input_im, image_col * channels + 0, image_row),
			tex2D<unsigned char>(input_im, image_col * channels + 1, image_row),
			tex2D<unsigned char>(input_im, image_col * channels + 2, image_row)
			);

		int scan_length, diff_curr_ref, diff_curr_next;

		//Upward scan
		scan_length = 0; diff_curr_ref = 0; diff_curr_next = 0;
		while (true)
		{
			uchar3 curr = make_uchar3(
				tex2D<unsigned char>(input_im, image_col * channels + 0, image_row - scan_length),
				tex2D<unsigned char>(input_im, image_col * channels + 1, image_row - scan_length),
				tex2D<unsigned char>(input_im, image_col * channels + 2, image_row - scan_length)
				);

			uchar3 next = make_uchar3(
				tex2D<unsigned char>(input_im, image_col * channels + 0, image_row - scan_length - 1),
				tex2D<unsigned char>(input_im, image_col * channels + 1, image_row - scan_length - 1),
				tex2D<unsigned char>(input_im, image_col * channels + 2, image_row - scan_length - 1)
				);

			diff_curr_ref = uchar3_max_diff(curr, ref);
			diff_curr_next = uchar3_max_diff(curr, next);

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
			uchar3 curr = make_uchar3(
				tex2D<unsigned char>(input_im, image_col * channels + 0, image_row + scan_length),
				tex2D<unsigned char>(input_im, image_col * channels + 1, image_row + scan_length),
				tex2D<unsigned char>(input_im, image_col * channels + 2, image_row + scan_length)
				);

			uchar3 next = make_uchar3(
				tex2D<unsigned char>(input_im, image_col * channels + 0, image_row + scan_length + 1),
				tex2D<unsigned char>(input_im, image_col * channels + 1, image_row + scan_length + 1),
				tex2D<unsigned char>(input_im, image_col * channels + 2, image_row + scan_length + 1)
				);

			diff_curr_ref = uchar3_max_diff(curr, ref);
			diff_curr_next = uchar3_max_diff(curr, next);

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
			uchar3 curr = make_uchar3(
				tex2D<unsigned char>(input_im, (image_col - scan_length) * channels + 0, image_row),
				tex2D<unsigned char>(input_im, (image_col - scan_length) * channels + 1, image_row),
				tex2D<unsigned char>(input_im, (image_col - scan_length) * channels + 2, image_row)
				);

			uchar3 next = make_uchar3(
				tex2D<unsigned char>(input_im, (image_col - scan_length - 1) * channels + 0, image_row),
				tex2D<unsigned char>(input_im, (image_col - scan_length - 1) * channels + 1, image_row),
				tex2D<unsigned char>(input_im, (image_col - scan_length - 1) * channels + 2, image_row)
				);

			diff_curr_ref = uchar3_max_diff(curr, ref);
			diff_curr_next = uchar3_max_diff(curr, next);

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
			uchar3 curr = make_uchar3(
				tex2D<unsigned char>(input_im, (image_col + scan_length) * channels + 0, image_row),
				tex2D<unsigned char>(input_im, (image_col + scan_length) * channels + 1, image_row),
				tex2D<unsigned char>(input_im, (image_col + scan_length) * channels + 2, image_row)
				);

			uchar3 next = make_uchar3(
				tex2D<unsigned char>(input_im, (image_col + scan_length + 1) * channels + 0, image_row),
				tex2D<unsigned char>(input_im, (image_col + scan_length + 1) * channels + 1, image_row),
				tex2D<unsigned char>(input_im, (image_col + scan_length + 1) * channels + 2, image_row)
				);

			diff_curr_ref = uchar3_max_diff(curr, ref);
			diff_curr_next = uchar3_max_diff(curr, next);

			if (!(scan_length < max_arm_length &&
				image_col + scan_length < width &&
				diff_curr_ref <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold) &&
				diff_curr_next <= (arm_length < scan_length ? strict_arm_threshold : arm_threshold))) break;

			scan_length++;
		}
		pix_arm.w = scan_length;

		pix_arm.x = pix_arm.x == 0 ? (image_row - 2 >= 0 ? 2 : 0) : pix_arm.x;
		pix_arm.y = pix_arm.y == 0 ? (image_row + 2 < height ? 2 : 0) : pix_arm.y;
		pix_arm.z = pix_arm.z == 0 ? (image_col - 2 >= 0 ? 2 : 0) : pix_arm.z;
		pix_arm.w = pix_arm.w == 0 ? (image_col + 2 < width ? 2 : 0) : pix_arm.w;

		ouput_arm_vol[image_row * width + image_col] = pix_arm;
	}
}

__global__
void cost_initialization_kernel(cudaTextureObject_t left_tex, cudaTextureObject_t left_census_tex, cudaTextureObject_t right_tex, cudaTextureObject_t right_census_tex, float4 *cost_vol, float ad_gamma, float census_gamma, bool left_to_right, int width, int height){
	//16 x 16 threads
	extern __shared__ float4 cost_cache[];

	int image_row = blockIdx.y;
	float4 accum = { 0.0f, 0.0f, 0.0f, 0.0f };

	for (int col_iter = 0, col_iter_lim = DIVIDE_UP(width, blockDim.y); col_iter < col_iter_lim; col_iter++){

		int image_col = __mul24(col_iter, blockDim.y) + threadIdx.y;

		if (image_row < height && image_col < width){

			//Compute costs
			float4 ad_costs, census_costs;
			unsigned int disps[4] = { blockDim.x * 0 + threadIdx.x + 1, blockDim.x * 1 + threadIdx.x + 1, blockDim.x * 2 + threadIdx.x + 1, blockDim.x * 3 + threadIdx.x + 1 };

			if (left_to_right){
				ad_costs = make_float4(
					(fabsf(tex2D<unsigned char>(left_tex, image_col, image_row) - tex2D<unsigned char>(right_tex, image_col - disps[0], image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col, image_row) - tex2D<unsigned char>(right_tex, image_col - disps[1], image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col, image_row) - tex2D<unsigned char>(right_tex, image_col - disps[2], image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col, image_row) - tex2D<unsigned char>(right_tex, image_col - disps[3], image_row)) / 255.0f) * ad_gamma
					);
				census_costs = make_float4(
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col, image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col - disps[0], image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col, image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col - disps[1], image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col, image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col - disps[2], image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col, image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col - disps[3], image_row)))) / 6.0f) * census_gamma
					);
			}
			else{
				ad_costs = make_float4(
					(fabsf(tex2D<unsigned char>(left_tex, image_col + disps[0], image_row) - tex2D<unsigned char>(right_tex, image_col, image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col + disps[1], image_row) - tex2D<unsigned char>(right_tex, image_col, image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col + disps[2], image_row) - tex2D<unsigned char>(right_tex, image_col, image_row)) / 255.0f) * ad_gamma,
					(fabsf(tex2D<unsigned char>(left_tex, image_col + disps[3], image_row) - tex2D<unsigned char>(right_tex, image_col, image_row)) / 255.0f) * ad_gamma
					);
				census_costs = make_float4(
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col + disps[0], image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col, image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col + disps[1], image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col, image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col + disps[2], image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col, image_row)))) / 6.0f) * census_gamma,
					(__popcll((int2_to_ll(tex2D<uint2>(left_census_tex, image_col + disps[3], image_row)) ^ int2_to_ll(tex2D<uint2>(right_census_tex, image_col, image_row)))) / 6.0f) * census_gamma
					);
			}

			//Compute the total cost

			cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x] = add_float4(ad_costs, census_costs);
			__syncthreads();

			//Compute prefix sum
			float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for (int i = 0; i <= threadIdx.y; i++){
				sum = add_float4(sum, cost_cache[__mul24(i, blockDim.x) + threadIdx.x]);
			}
			__syncthreads();

			//Write to global memory
			cost_vol[threadIdx.x + __mul24(image_col, blockDim.x) + __mul24(image_row, __mul24(width, blockDim.x))] = add_float4(sum, accum);

			//Update cache
			if (threadIdx.y == blockDim.y - 1) cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x] = sum;
			__syncthreads();

			//Update accumulator
			accum = add_float4(accum, cost_cache[__mul24(blockDim.y - 1, blockDim.x) + threadIdx.x]);
		}
	}
}

__global__
void horizontal_aggregation_kernel(float4 *cost_vol_in, uchar4 *arm_vol, float4 *cost_vol_out, int width, int height, int max_disparity){
	extern __shared__ float4 cost_cache[];

	int image_col = blockIdx.x;

	float4 accum = { 0.0f, 0.0f, 0.0f, 0.0f };

	int prev_right_lim = -1;
	int prev_left_lim = -1;
	float4 right_lim_temp;
	float4 left_lim_temp;

	for (int row_iter = 0, row_iter_lim = DIVIDE_UP(height, blockDim.y); row_iter < row_iter_lim; row_iter++){

		int image_row = __mul24(row_iter, blockDim.y) + threadIdx.y;

		if (image_row < height && image_col < width){

			//Load arm data
			uchar4 pix_arm = arm_vol[image_col + image_row * width];
			int right_lim = image_col + pix_arm.w; int left_lim = image_col - pix_arm.z - 1;

			//Aggregate
			float4 aggregate;
			if (prev_right_lim != right_lim){
				right_lim_temp = cost_vol_in[threadIdx.x + __mul24(right_lim, blockDim.x) + __mul24(image_row, __mul24(width, blockDim.x))];
				prev_right_lim = right_lim;
			}

			if (left_lim >= 0 && prev_left_lim != left_lim){
				left_lim_temp = cost_vol_in[threadIdx.x + __mul24(left_lim, blockDim.x) + __mul24(image_row, __mul24(width, blockDim.x))];
				prev_left_lim = left_lim;
				aggregate = subtract_float4(aggregate, left_lim_temp);
			}

			cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x] = aggregate;
			__syncthreads();

			//Compute prefix sum
			float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for (int i = 0; i <= threadIdx.y; i++){ sum = add_float4(sum, cost_cache[__mul24(i, blockDim.x) + threadIdx.x]); }
			__syncthreads();

			//Write to global memory
			cost_vol_out[threadIdx.x + __mul24(image_col, blockDim.x) + __mul24(image_row, __mul24(width, blockDim.x))] = add_float4(sum, accum);

			//Update cache
			if (threadIdx.y == blockDim.y - 1) cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x] = sum;
			__syncthreads();

			//Update accumulator
			accum = add_float4(accum, cost_cache[__mul24(blockDim.y - 1, blockDim.x) + threadIdx.x]);
		}
	}
}

__global__
void vertical_aggregation_kernel(float4 *cost_vol_in, uchar4 *arm_vol, float4 *cost_vol_out, unsigned char *disp_im, int width, int height, int max_disparity){
	extern __shared__ float4 cost_cache[];

	int image_row = blockIdx.y;

	for (int col_iter = 0, col_iter_lim = DIVIDE_UP(width, blockDim.y); col_iter < col_iter_lim; col_iter++){

		int image_col = __mul24(col_iter, blockDim.y) + threadIdx.y;

		if (image_row < height && image_col < width){

			//Load arm data
			uchar4 pix_arm = arm_vol[image_col + __mul24(image_row, width)];

			int down_lim = image_row + pix_arm.y;
			int up_lim = image_row - pix_arm.x - 1;

			//Aggregate
			float4 aggregate = cost_vol_in[threadIdx.x + __mul24(image_col, blockDim.x) + __mul24(down_lim, __mul24(width, blockDim.x))];

			if (up_lim >= 0){
				float4 up_lim_temp = cost_vol_in[threadIdx.x + __mul24(image_col, blockDim.x) + __mul24(up_lim, __mul24(width, blockDim.x))];
				aggregate = subtract_float4(aggregate, up_lim_temp);
			}

			//Find the local minimum
			bool invalid = false;

			unsigned int disps[4] = { blockDim.x * 0 + threadIdx.x + 1, blockDim.x * 1 + threadIdx.x + 1, blockDim.x * 2 + threadIdx.x + 1, blockDim.x * 3 + threadIdx.x + 1 };
			float min_cost = aggregate.x; float min_disp = disps[0];

			if (aggregate.y < min_cost){ min_cost = aggregate.y; min_disp = disps[1]; }
			else if (aggregate.y == min_cost) { invalid = true; }
			else{}

			if (aggregate.z < min_cost){ min_cost = aggregate.z; min_disp = disps[2]; }
			else if (aggregate.z == min_cost) { invalid = true; }
			else{}

			if (aggregate.w < min_cost){ min_cost = aggregate.w; min_disp = disps[3]; }
			else if (aggregate.w == min_cost) { invalid = true; }
			else{}

			cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x = min_cost;
			cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].y = min_disp;
			__syncthreads();

			//Find the global minimum
			int thread_selector = 2; int stride = 1;

			for (int reduce_iter = 0, reduce_iter_lim = (int)ceil(log2f(blockDim.x)) - 1; reduce_iter < reduce_iter_lim; reduce_iter++){
				if (threadIdx.x % thread_selector == 0){
					if (((threadIdx.x + stride) >= blockDim.x ? INFINITY : cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x) < cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x){
						cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x = cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x;
						cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].y = cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].y;
					}
					else if (((threadIdx.x + stride) >= blockDim.x ? INFINITY : cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x) == cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x) invalid = true;
					else{}
				}
				__syncthreads();

				thread_selector = thread_selector << 1; stride = stride << 1;
			}

			if (threadIdx.x % thread_selector == 0){
				if (((threadIdx.x + stride) >= blockDim.x ? INFINITY : cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x) < cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x){
					cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x = cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x;
					cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].y = cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].y;
				}
				else if (((threadIdx.x + stride) >= blockDim.x ? INFINITY : cost_cache[__mul24(threadIdx.y, blockDim.x) + (threadIdx.x + stride)].x) == cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].x) invalid = true;
				else{}
				disp_im[image_col + __mul24(image_row, width)] = invalid ? OUTLIER : (char)cost_cache[__mul24(threadIdx.y, blockDim.x) + threadIdx.x].y;
			}
		}
	}
}

__global__
void consistency_check_kernel(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned char *output_disp_im, int disparity_tolerance, int width, int height){

	int col_to_access = blockIdx.x * blockDim.x + threadIdx.x;
	int row_to_access = blockIdx.y * blockDim.y + threadIdx.y;

	if (row_to_access < height && col_to_access < width){
		unsigned char disp = tex2D<unsigned char>(left_disp_im, col_to_access, row_to_access);
		unsigned char to_check = tex2D<unsigned char>(right_disp_im, col_to_access - disp, row_to_access);

		output_disp_im[row_to_access * width + col_to_access] = (abs(disp - to_check) <= disparity_tolerance) ? disp : OUTLIER;
	}
}

__global__
void horizontal_voting_kernel(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){

		int sums[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		int eligible_votes = 0;
		int no_of_votes = 0;

		//Load arm data
		uchar4 pix_arm = arm_vol[image_col + image_row * width];

		//Check disp value
		int disp_value = tex2D<unsigned char>(input_disp, image_col, image_row);

		if (disp_value == OUTLIER){
			for (int pix_iter = -pix_arm.z; pix_iter <= pix_arm.w; pix_iter++){
				int disp_val = tex2D<unsigned char>(input_disp, image_col + pix_iter, image_row);
				if (disp_val != OUTLIER){
					sums[0] += ((disp_val & 1) != 0);
					sums[1] += ((disp_val & 2) != 0);
					sums[2] += ((disp_val & 4) != 0);
					sums[3] += ((disp_val & 8) != 0);
					sums[4] += ((disp_val & 16) != 0);
					sums[5] += ((disp_val & 32) != 0);
					sums[6] += ((disp_val & 64) != 0);
					sums[7] += ((disp_val & 128) != 0);
					eligible_votes++;
				}
				no_of_votes++;
			}
			__syncthreads();

			int majority = eligible_votes * 0.5;
			disp_value = (((sums[7] > majority) << 7) +
				((sums[6] > majority) << 6) +
				((sums[5] > majority) << 5) +
				((sums[4] > majority) << 4) +
				((sums[3] > majority) << 3) +
				((sums[2] > majority) << 2) +
				((sums[1] > majority) << 1) +
				((sums[0] > majority) << 0));
			disp_value = (eligible_votes > no_of_votes * 0.65f) ? disp_value : OUTLIER;
		}

		output_disp[image_col + image_row * width] = disp_value;
	}//bounds check
}//kernel

__global__
void vertical_voting_kernel(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height){
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;

	if (image_row < height && image_col < width){

		int sums[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		int eligible_votes = 0;
		int no_of_votes = 0;

		//Load arm data
		uchar4 pix_arm = arm_vol[image_col + image_row * width];

		//Check disp value
		int disp_value = tex2D<unsigned char>(input_disp, image_col, image_row);

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
					eligible_votes++;
				}
				no_of_votes++;
			}
			__syncthreads();

			int majority = eligible_votes * 0.5;
			disp_value = (((sums[7] > majority) << 7) +
				((sums[6] > majority) << 6) +
				((sums[5] > majority) << 5) +
				((sums[4] > majority) << 4) +
				((sums[3] > majority) << 3) +
				((sums[2] > majority) << 2) +
				((sums[1] > majority) << 1) +
				((sums[0] > majority) << 0));
			disp_value = (eligible_votes > no_of_votes * 0.65f) ? disp_value : OUTLIER;
		}

		output_disp[image_col + image_row * width] = disp_value;
	}//bounds check
}//kernel

__global__
void extrapolation_kernel(cudaTextureObject_t input_disp, unsigned char *output_disp, int width, int height){
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;

	if (image_row < height && image_col < width){
		if (tex2D<unsigned char>(input_disp, image_col, image_row) == OUTLIER){
			int left_disp = OUTLIER, right_disp = OUTLIER;

			int gap = 0;
			bool is_left_edge = false, is_right_edge = false;

			//Scan leftward
			for (int i = image_col; i >= 0; i--){
				if (tex2D<unsigned char>(input_disp, i, image_row) == OUTLIER){
					gap++;
					continue;
				}
				else{
					is_left_edge = i <= 0;
					left_disp = tex2D<unsigned char>(input_disp, i, image_row);
					break;
				}
			}

			//Scan rightward
			for (int i = image_col; i < width; i++){
				if (tex2D<unsigned char>(input_disp, i, image_row) == OUTLIER){
					gap++;
					continue;
				}
				else{
					is_right_edge = i >= (width - 1);
					right_disp = tex2D<unsigned char>(input_disp, i, image_row);
					break;
				}

			}

			int val = min(is_left_edge ? right_disp : left_disp, is_right_edge ? left_disp : right_disp);
			val = gap <= 40 ? val : ((is_right_edge || is_left_edge) ? val : OUTLIER);
			output_disp[image_col + image_row * width] = val;
		}
	}
}

__global__
void cleanup_kernel(unsigned char *input_disp, unsigned char *output_disp, int width, int height, int min_disparity){
	int image_col = blockIdx.x * blockDim.x + threadIdx.x;
	int image_row = blockIdx.y * blockDim.y + threadIdx.y;

	if (image_row < height && image_col < width){
		int disp = input_disp[image_col + image_row * width];
		if (disp == OUTLIER || disp < min_disparity){
			output_disp[image_col + image_row * width] = 0;
		}
	}
}


/////////////////////////////////////////////////////////////////////////////Stubs/////////////////////////////////////////////////////////////////////////////

void census_transform(cudaTextureObject_t input_im, uint2 *output_census, int width, int height, cudaStream_t stream){
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

void cross_construct_bgr(cudaTextureObject_t input_im, uchar4 *arm_vol, int arm_length, int max_arm_length, int arm_threshold, int strict_arm_threshold, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	cross_construct_kernel_bgr << <blocks, threads >> >(input_im, arm_vol, arm_length, max_arm_length, arm_threshold, strict_arm_threshold, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Cross construct BGR failed.");
#endif
}

void match(cudaTextureObject_t left_tex, cudaTextureObject_t left_census_tex, cudaTextureObject_t right_tex, cudaTextureObject_t right_census_tex, float4 *cost_vol_temp_a, float4 *cost_vol_temp_b, uchar4 *arm_vol, unsigned char *disp_im, float ad_gamma, float census_gamma, bool left_to_right, int width, int height, int max_disparity, cudaStream_t stream){

	dim3 threads;
	threads.x = max_disparity / GRANULARITY;
	threads.y = max_disparity >= 32 ? 16 : 32;

	dim3 blocks(1, height);

	size_t smem_sz = threads.x * threads.y * sizeof(float) * GRANULARITY;
	cost_initialization_kernel << <blocks, threads, smem_sz >> >(left_tex, left_census_tex, right_tex, right_census_tex, cost_vol_temp_a, ad_gamma, census_gamma, left_to_right, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Cost initialization failed.");
#endif

	blocks.x = width; blocks.y = 1;
	horizontal_aggregation_kernel << <blocks, threads, smem_sz >> >(cost_vol_temp_a, arm_vol, cost_vol_temp_b, width, height, max_disparity);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal aggregation failed.");
#endif

	blocks.x = 1; blocks.y = height;
	vertical_aggregation_kernel << <blocks, threads, smem_sz >> >(cost_vol_temp_b, arm_vol, cost_vol_temp_a, disp_im, width, height, max_disparity);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal aggregation failed.");
#endif
}

void check_consistency(cudaTextureObject_t left_disp_im, cudaTextureObject_t right_disp_im, unsigned char *output_disp_im, int disparity_tolerance, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	consistency_check_kernel << <blocks, threads >> >(left_disp_im, right_disp_im, output_disp_im, disparity_tolerance, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Consistency check failed.");
#endif
}

void horizontal_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	horizontal_voting_kernel << <blocks, threads >> >(input_disp, arm_vol, output_disp, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Horizontal voting failed.");
#endif
}

void vertical_voting(cudaTextureObject_t input_disp, uchar4 *arm_vol, unsigned char *output_disp, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	vertical_voting_kernel << <blocks, threads >> >(input_disp, arm_vol, output_disp, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Vertical voting failed.");
#endif
}

void extrapolation(cudaTextureObject_t input_disp, unsigned char *output_disp, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	extrapolation_kernel << <blocks, threads >> >(input_disp, output_disp, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Extrapolation failed.");
#endif
}

void color2gray(cudaTextureObject_t input, unsigned char *output, int width, int height, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	color2gray_kernel << <blocks, threads >> >(input, output, width, height);
#ifdef KERN_DEB
	SAFE_CALL(cudaDeviceSynchronize(), "Color conversion failed.");
#endif
}

void cleanup(unsigned char *input, unsigned char *output, int width, int height, int min_disparity, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 blocks(DIVIDE_UP(width, threads.x), DIVIDE_UP(height, threads.y));

	cleanup_kernel << <blocks, threads >> >(input, output, width, height, min_disparity);
}