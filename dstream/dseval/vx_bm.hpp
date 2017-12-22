#pragma once
#include <VX/vx.h>

class NVXBM
{
public:
	struct NVXBMParams
	{
		// disparity range
		vx_int32 min_disparity;
		vx_int32 max_disparity;

		// SAD window size
		vx_int32 sad;

		NVXBMParams();
	};

	static NVXBM* createNVXBM(vx_context context, const NVXBMParams& params,
		vx_image left, vx_image right, vx_image disparity);

	virtual ~NVXBM() {}

	virtual void run() = 0;

	virtual float getPerf() = 0;
};
