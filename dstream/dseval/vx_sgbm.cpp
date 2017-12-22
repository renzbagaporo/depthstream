/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "vx_sgbm.hpp"

#include <climits>
#include <cfloat>
#include <iostream>
#include <iomanip>

#include <VX/vxu.h>
#include <NVX/nvx.h>

#include <NVXIO/Utility.hpp>
#include "NVXIO/Application.hpp"

#define SCALE 1

namespace
{
	//
	// SGM-based stereo matching
	//

	class SGBM : public NVXSGBM
	{
	public:
		SGBM(vx_context context, const NVXSGBMParams& params,
			vx_image left, vx_image right, vx_image disparity);
		~SGBM();

		virtual void run();

		float getPerf();

	private:
		vx_graph main_graph_;
		vx_node left_downscale_node_;
		vx_node right_downscale_node_;
		vx_node semi_global_matching_node_;
		vx_node convert_depth_node_;
		vx_node disparity_upscale_node_;
		vx_node disparity_multiply_node_;
	};

	void SGBM::run()
	{
		NVXIO_SAFE_CALL(vxProcessGraph(main_graph_));
	}

	SGBM::~SGBM()
	{
		vxReleaseGraph(&main_graph_);
	}

	SGBM::SGBM(vx_context context, const NVXSGBMParams& params,
		vx_image left, vx_image right, vx_image disparity)
		: main_graph_(nullptr)
	{
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_uint32 full_width = 0;
		vx_uint32 full_height = 0;

		NVXIO_SAFE_CALL(vxQueryImage(left, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		NVXIO_SAFE_CALL(vxQueryImage(left, VX_IMAGE_ATTRIBUTE_WIDTH, &full_width, sizeof(full_width)));
		NVXIO_SAFE_CALL(vxQueryImage(left, VX_IMAGE_ATTRIBUTE_HEIGHT, &full_height, sizeof(full_height)));

		vx_uint8 disparity_scale = SCALE;
		vx_uint32 width = (full_width / disparity_scale / 2) * 2; // guarantee, that width % 2 == 0
		vx_uint32 height = (full_height / disparity_scale / 2) * 2;

		main_graph_ = vxCreateGraph(context);
		NVXIO_CHECK_REFERENCE(main_graph_);

		// downscale images before evaluating stereo
		vx_image left_downscaled = vxCreateVirtualImage(main_graph_, width, height, format);
		NVXIO_CHECK_REFERENCE(left_downscaled);

		vx_image right_downscaled = vxCreateVirtualImage(main_graph_, width, height, format);
		NVXIO_CHECK_REFERENCE(right_downscaled);

		left_downscale_node_ = vxScaleImageNode(main_graph_, left, left_downscaled, VX_INTERPOLATION_TYPE_BILINEAR);
		NVXIO_CHECK_REFERENCE(left_downscale_node_);

		right_downscale_node_ = vxScaleImageNode(main_graph_, right, right_downscaled, VX_INTERPOLATION_TYPE_BILINEAR);
		NVXIO_CHECK_REFERENCE(right_downscale_node_);

		// evaluate stereo
		vx_image disparity_short = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_S16);
		NVXIO_CHECK_REFERENCE(disparity_short);


		semi_global_matching_node_ = nvxSemiGlobalMatchingNode(
			main_graph_,
			left_downscaled,
			right_downscaled,
			disparity_short,
			params.min_disparity,
			params.max_disparity,
			params.P1,
			params.P2,
			params.sad,
			params.bt_clip_value,
			params.max_diff,
			params.uniqueness_ratio,
			params.scanlines_mask);

		NVXIO_CHECK_REFERENCE(semi_global_matching_node_);

		// convert disparity from fixed point to grayscale
		vx_int32 shift = 4;
		vx_scalar s_shift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
		NVXIO_CHECK_REFERENCE(s_shift);

		vx_image disparity_downscaled = vxCreateVirtualImage(main_graph_, width, height, VX_DF_IMAGE_U8);
		NVXIO_CHECK_REFERENCE(disparity_downscaled);

		convert_depth_node_ = vxConvertDepthNode(main_graph_, disparity_short, disparity_downscaled, VX_CONVERT_POLICY_SATURATE, s_shift);
		NVXIO_CHECK_REFERENCE(convert_depth_node_);

		// upscale disparity to the size of the original images
		vx_image disparity_upscaled = vxCreateVirtualImage(main_graph_, full_width, full_height, VX_DF_IMAGE_U8);
		NVXIO_CHECK_REFERENCE(disparity_upscaled);

		disparity_upscale_node_ = vxScaleImageNode(main_graph_, disparity_downscaled, disparity_upscaled, VX_INTERPOLATION_TYPE_BILINEAR);
		NVXIO_CHECK_REFERENCE(disparity_upscale_node_);

		// multiply with scaling factor after upscaling to maintain correctness
		// (in case of scaling factor of 2, this step can be merged with
		// ConvertDepth step)
		vx_image disparity_multiplier = vxCreateUniformImage(context, full_width, full_height, VX_DF_IMAGE_U8, &disparity_scale);
		NVXIO_CHECK_REFERENCE(disparity_multiplier);

		vx_float32 scale = 1;
		vx_scalar s_scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &scale);

		disparity_multiply_node_ = vxMultiplyNode(main_graph_, disparity_upscaled, disparity_multiplier, s_scale,
			VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, disparity);
		NVXIO_CHECK_REFERENCE(disparity_multiply_node_);

		// verify the graph
		NVXIO_SAFE_CALL(vxVerifyGraph(main_graph_));

		// clean up
		vxReleaseScalar(&s_shift);
		vxReleaseScalar(&s_scale);

		vxReleaseImage(&left_downscaled);
		vxReleaseImage(&right_downscaled);

		vxReleaseImage(&disparity_short);
		vxReleaseImage(&disparity_downscaled);
		vxReleaseImage(&disparity_upscaled);
		vxReleaseImage(&disparity_multiplier);
	}

	float SGBM::getPerf()
	{
		vx_perf_t perf;
		NVXIO_SAFE_CALL(vxQueryNode(semi_global_matching_node_, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)));
		return (float)(perf.tmp / 1000000.0);
	}
}

NVXSGBM* NVXSGBM::createNVXSGBM(vx_context context, const NVXSGBMParams& params,
	vx_image left, vx_image right, vx_image disparity)
{
	return new SGBM(context, params, left, right, disparity);
}

NVXSGBM::NVXSGBMParams::NVXSGBMParams()
{
	min_disparity = 0;
	max_disparity = 64;
	P1 = 8;
	P2 = 109;
	sad = 5;
	bt_clip_value = 31;
	max_diff = 32000;
	uniqueness_ratio = 0;
	scanlines_mask = 255;
}
