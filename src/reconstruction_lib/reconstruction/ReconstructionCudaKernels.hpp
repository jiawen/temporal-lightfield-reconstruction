/*
 *  Copyright (c) 2010-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-------------------------------------------------------------------------------------------------
// CUDA reconstruction kernel headers
//-------------------------------------------------------------------------------------------------

#pragma once
#include "base/DLLImports.hpp"
#include "base/Math.hpp"

namespace FW
{

struct CudaSample
{
	float	x, y;
	float	u, v, t;
};

struct CudaNode
{
#if FW_CUDA
	__device__
#endif
	bool	isLeaf() const { return idx0 <= idx1; }

	Vec3f	planes[4];		// time+lens hyperplane bounds xmin,xmax,ymin,ymax
	Vec2f	tspan;
	int		idx0, idx1;
};

struct CudaTraversalNode
{
	float4	planes0;
	float4	planes1;
	float4	planes2;
	float4	coc_idx;
};

struct CudaPoint
{
	float	w, t, mvw, pad;
	float   x, y, mvx, mvy;
	Vec4f	c; // RGBA
};

//-------------------------------------------------------------------------------------------------

struct RecKernelInput
{
	CUdeviceptr		osmp;		// output sample templates
	CUdeviceptr		nodes;
	CUdeviceptr		points;
	CUdeviceptr		nearest;
	CUdeviceptr		failed;
	CUdeviceptr		overflow;
	CUdeviceptr		resultImg;
	float2			cocCoeffs;
	int2			size;
	U32				firstSample;
	U32				numSamples;
	float			dispersion;
	float			spp;
	U32				rootNode;
	U32				outputSpp;
	U32				numPatterns;
};

#if FW_CUDA
extern "C"
{
	__device__   S32            g_failedCount;
	__device__   S32            g_nearestCount;
	__device__   S32            g_overflowCount;
	__constant__ RecKernelInput c_RecKernelInput;
	texture<float4, 1>  t_nodes;
	texture<float4, 1>  t_points;
}
#endif

//-------------------------------------------------------------------------------------------------
}
