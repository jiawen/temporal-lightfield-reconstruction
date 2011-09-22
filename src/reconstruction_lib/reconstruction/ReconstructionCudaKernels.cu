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
// CUDA reconstruction kernels
//-------------------------------------------------------------------------------------------------

#include "ReconstructionCudaKernels.hpp"

// EMIT_NVCC_OPTIONS -m32 -use_fast_math

using namespace FW;

enum {
	MAX_POINTS		= 128,		// maximum number of points we can handle on one surface
	MAX_LEAVES_R1	= 64,		// max leaves for first pass
	MAX_LEAVES_R2	= 64,		// max leaves for second (fail) pass
};

//------------------------------------------------------------------------

__device__ __inline__ U32   getLo                   (U64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ S32   getLo                   (S64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ U32   getHi                   (U64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ S32   getHi                   (S64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ U64   combineLoHi             (U32 lo, U32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ S64   combineLoHi             (S32 lo, S32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskLe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGe           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }
__device__ __inline__ int   findLeadingOne          (U32 v)                 { U32 r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
__device__ __inline__ int   findLastOne             (U32 v)                 { return 31-findLeadingOne(__brev(v)); } // 0..31, 32 for not found
__device__ __inline__ bool  singleLane              (void)                  { return ((__ballot(true) & getLaneMaskLt()) == 0); }
__device__ __inline__ U32   rol						(U32 x, U32 s)			{ return (x<<s)|(x>>(32-s)); }
__device__ __inline__ Vec2f U64toVec2f				(U64 xy)				{ return Vec2f(__int_as_float(getLo(xy)), __int_as_float(getHi(xy))); }
__device__ __inline__ U64   Vec2ftoU64				(const Vec2f& v)		{ return combineLoHi(__float_as_int(v.x), __float_as_int(v.y)); }
__device__ __inline__ int   imin					(S32 a, S32 b)			{ S32 v; asm("min.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ int   imax					(S32 a, S32 b)			{ S32 v; asm("max.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }

//------------------------------------------------------------------------

__device__ inline bool intersectNodeSample(const CudaNode& node, const CudaSample& s, float radius)
{
	Vec2f mn = Vec2f(dot(node.planes[0], Vec3f(s.t, s.u, 1.0f)), dot(node.planes[2], Vec3f(s.t, s.v, 1.0f))) - radius;
	Vec2f mx = Vec2f(dot(node.planes[1], Vec3f(s.t, s.u, 1.0f)), dot(node.planes[3], Vec3f(s.t, s.v, 1.0f))) + radius;
	return (s.x >= mn.x && s.y >= mn.y && s.x <= mx.x && s.y <= mx.y);
}

//------------------------------------------------------------------------

__device__ inline bool intersectNodeSample2(const float4& planes0, const float4& planes1, const float4& planes2,
											float u, float v, float t, float x, float y, float radius)
{
	float minx = planes0.x * t + planes0.y * u + planes0.z - radius;
	float maxx = planes0.w * t + planes1.x * u + planes1.y + radius;
	float miny = planes1.z * t + planes1.w * v + planes2.x - radius;
	float maxy = planes2.y * t + planes2.z * v + planes2.w + radius;
	return (x >= minx && x <= maxx && y >= miny && y <= maxy);
}

//-------------------------------------------------------------------------------------------------

__device__ inline int ternaryCompare( float a, float b, float eps )
{
	if ( fabs(a-b) < eps )
		return 0;
	else if ( a < b )
		return -1;
	else
		return 1;
}

__device__ inline bool ternaryEqual( int a, int b )
{
	return a==0 || b==0 || a==b;
}

__device__ inline bool sameSurface(const CudaNode& n0, const CudaNode& n1, const Vec3f& uvtmin, const Vec3f& uvtmax)
{
	const float COC_THRESHOLD = 1*0.1f;
	float tmid = 0.5f * (uvtmin.z+uvtmax.z);
	
	float Xmin00 = dot( n0.planes[0], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmin10 = dot( n0.planes[0], Vec3f( tmid, uvtmax.x, 1.0f ) );
	float Xmax00 = dot( n0.planes[1], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmax10 = dot( n0.planes[1], Vec3f( tmid, uvtmax.x, 1.0f ) );

	float Ymin00 = dot( n0.planes[2], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymin10 = dot( n0.planes[2], Vec3f( tmid, uvtmax.y, 1.0f ) );
	float Ymax00 = dot( n0.planes[3], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymax10 = dot( n0.planes[3], Vec3f( tmid, uvtmax.y, 1.0f ) );

	float Xmin01 = dot( n1.planes[0], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmin11 = dot( n1.planes[0], Vec3f( tmid, uvtmax.x, 1.0f ) );
	float Xmax01 = dot( n1.planes[1], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmax11 = dot( n1.planes[1], Vec3f( tmid, uvtmax.x, 1.0f ) );

	float Ymin01 = dot( n1.planes[2], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymin11 = dot( n1.planes[2], Vec3f( tmid, uvtmax.y, 1.0f ) );
	float Ymax01 = dot( n1.planes[3], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymax11 = dot( n1.planes[3], Vec3f( tmid, uvtmax.y, 1.0f ) );

	bool bCoC =
		   ternaryEqual( ternaryCompare( Xmin00, Xmin01, COC_THRESHOLD ), ternaryCompare( Xmin10, Xmin11, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( Ymin00, Ymin01, COC_THRESHOLD ), ternaryCompare( Ymin10, Ymin11, COC_THRESHOLD ) ) && 
		   ternaryEqual( ternaryCompare( Xmax00, Xmax01, COC_THRESHOLD ), ternaryCompare( Xmax10, Xmax11, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( Ymax00, Ymax01, COC_THRESHOLD ), ternaryCompare( Ymax10, Ymax11, COC_THRESHOLD ) );

	if ( !bCoC )
		return false;
						 
	// evaluate XY (without UV) bounds at ta,tb
	float xmin_ta_node0 = uvtmin.z * n0.planes[0].x + n0.planes[0].z;
	float xmin_tb_node0 = uvtmax.z * n0.planes[0].x + n0.planes[0].z;
	float xmax_ta_node0 = uvtmin.z * n0.planes[1].x + n0.planes[1].z;
	float xmax_tb_node0 = uvtmax.z * n0.planes[1].x + n0.planes[1].z;
	float ymin_ta_node0 = uvtmin.z * n0.planes[2].x + n0.planes[2].z;
	float ymin_tb_node0 = uvtmax.z * n0.planes[2].x + n0.planes[2].z;
	float ymax_ta_node0 = uvtmin.z * n0.planes[3].x + n0.planes[3].z;
	float ymax_tb_node0 = uvtmax.z * n0.planes[3].x + n0.planes[3].z;
	float xmin_ta_node1 = uvtmin.z * n1.planes[0].x + n1.planes[0].z;
	float xmin_tb_node1 = uvtmax.z * n1.planes[0].x + n1.planes[0].z;
	float xmax_ta_node1 = uvtmin.z * n1.planes[1].x + n1.planes[1].z;
	float xmax_tb_node1 = uvtmax.z * n1.planes[1].x + n1.planes[1].z;
	float ymin_ta_node1 = uvtmin.z * n1.planes[2].x + n1.planes[2].z;
	float ymin_tb_node1 = uvtmax.z * n1.planes[2].x + n1.planes[2].z;
	float ymax_ta_node1 = uvtmin.z * n1.planes[3].x + n1.planes[3].z;
	float ymax_tb_node1 = uvtmax.z * n1.planes[3].x + n1.planes[3].z;

	bool bTime =
		   ternaryEqual( ternaryCompare( xmin_ta_node0, xmin_ta_node1, COC_THRESHOLD ), ternaryCompare( xmin_tb_node0, xmin_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( xmax_ta_node0, xmax_ta_node1, COC_THRESHOLD ), ternaryCompare( xmax_tb_node0, xmax_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( ymin_ta_node0, ymin_ta_node1, COC_THRESHOLD ), ternaryCompare( ymin_tb_node0, ymin_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( ymax_ta_node0, ymax_ta_node1, COC_THRESHOLD ), ternaryCompare( ymax_tb_node0, ymax_tb_node1, COC_THRESHOLD ) );

	return bTime;
}

//------------------------------------------------------------------------

#define addFailed(idx_) do {\
	int num = __popc(__ballot(true)); \
	if (singleLane()) \
		s_foo[threadIdx.y*32] = atomicAdd(&g_failedCount, num); \
	int pos = s_foo[threadIdx.y*32]; \
	pos += __popc(__ballot(true) & getLaneMaskLt()); \
	((U32*)in.failed)[pos] = (idx_); \
} while(0)

#define addNearest(idx_) do {\
	int num = __popc(__ballot(true)); \
	if (singleLane()) \
		s_foo[threadIdx.y*32] = atomicAdd(&g_nearestCount, num); \
	int pos = s_foo[threadIdx.y*32]; \
	pos += __popc(__ballot(true) & getLaneMaskLt()); \
	((U32*)in.nearest)[pos] = (idx_); \
} while(0)

#define addOverflow(idx_) do {\
	int num = __popc(__ballot(true)); \
	if (singleLane()) \
		s_foo[threadIdx.y*32] = atomicAdd(&g_overflowCount, num); \
	int pos = s_foo[threadIdx.y*32]; \
	pos += __popc(__ballot(true) & getLaneMaskLt()); \
	((U32*)in.overflow)[pos] = (idx_); \
} while(0)

#define storeResult(idx_,color_) do {\
	U32 pidx = ((idx_) + in.firstSample) / in.outputSpp; \
	float* cptr = (float*)(in.resultImg + (pidx << 4)); \
	atomicAdd(&cptr[0], (color_).x); \
	atomicAdd(&cptr[1], (color_).y); \
	atomicAdd(&cptr[2], (color_).z); \
	atomicAdd(&cptr[3], (color_).w); \
} while (0)

//------------------------------------------------------------------------

#define FW_HASH_MAGIC   (0x9e3779b9u)
#define FW_JENKINS_MIX(a, b, c)   \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15);

__device__ __inline__ U32  hashBits(U32 a, U32 b = FW_HASH_MAGIC, U32 c = 0)
{
	c += FW_HASH_MAGIC;
	FW_JENKINS_MIX(a, b, c);
	return c;
}

__device__ __inline__ void constructOutputSample(U32 id, CudaSample& s)
{
    const RecKernelInput& in = *(const RecKernelInput*)&c_RecKernelInput;
	id += in.firstSample; // this is the real sample index
	U32 y = id / (in.size.x * in.outputSpp);
	id -= y * in.size.x * in.outputSpp;
	U32 x = id / in.outputSpp;
	id -= x * in.outputSpp;
	U32 pattern = hashBits(x, y) % in.numPatterns;
	CudaSample& is = ((CudaSample*)in.osmp)[pattern * in.outputSpp+id];
	s.x = is.x + (float)x;
	s.y = is.y + (float)y;
	s.u = is.u;
	s.v = is.v;
	s.t = is.t;
}

//------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(128,7) recKernel(void)
{
	__shared__ volatile int s_foo[128];
    const RecKernelInput& in = *(const RecKernelInput*)&c_RecKernelInput;

    int tidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (tidx >= in.numSamples)
		return;

	CudaSample s;
	constructOutputSample(tidx, s);

	// collect points, only process first surface
	float radius = in.dispersion;
	enum { STACK_SIZE = 32 };
	U32 stack[STACK_SIZE];
	U64 leaves[MAX_LEAVES_R1];	// lo = idx, hi = key
	S32 stackPtr = 0;
	int numLeaves = 0;
	stack[0] = in.rootNode;

	float su = s.u;
	float sv = s.v;
	float st = s.t;
	float sx = s.x;
	float sy = s.y;
	while(stackPtr >= 0)
	{
		int idx = stack[stackPtr];
		stackPtr--;

		float4 planes0 = tex1Dfetch(t_nodes, idx*4+0);
		float4 planes1 = tex1Dfetch(t_nodes, idx*4+1);
		float4 planes2 = tex1Dfetch(t_nodes, idx*4+2);
		float4 coc_idx = tex1Dfetch(t_nodes, idx*4+3);
		if (intersectNodeSample2(planes0, planes1, planes2, su, sv, st, sx, sy, radius))
		{
			int idx0 = __float_as_int(coc_idx.z);
			int idx1 = __float_as_int(coc_idx.w);
			if (idx0 <= idx1)
			{
				if (numLeaves == MAX_LEAVES_R1)
				{
					numLeaves++;
					break;
				}

				const CudaPoint& p0 = ((const CudaPoint*)in.points)[idx0];
				float w = p0.w + (st - p0.t) * p0.mvw;
				leaves[numLeaves] = combineLoHi(idx, __float_as_int(w));
				numLeaves++;
			}
			else
			{
				stackPtr += 2;
				stack[stackPtr-1] = idx0;
				stack[stackPtr  ] = idx1;
			}
		}
	}

	// no leaves, retry again with double radius
	if (!numLeaves)
	{
		addFailed(tidx);
		return;
	}

	// too many leaves -> epic fail
	if (numLeaves > MAX_LEAVES_R1)
	{
		addOverflow(tidx);
		return;
	}

	// sort leaves
	for (int i=0; i < numLeaves-1; i++)
	{
		float w  = __int_as_float(getHi(leaves[i]));
		int   mn = i;
		for (int j=i+1; j < numLeaves; j++)
		{
			float w1 = __int_as_float(getHi(leaves[j]));
			if (w1 < w)
			{
				w  = w1;
				mn = j;
			}
		}
		if (mn != i)
			swap(leaves[i], leaves[mn]);
	}

	// reconstruct the color of first surface
	Vec4f wbox;
	bool  multi = false;
	Vec4f color(0.f, 0.f, 0.f, 0.f);
	int   lidx = 0;
	U64   empty = 0;
	wbox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
	for(; lidx < numLeaves; lidx++)
	{
		const CudaNode* node = &((const CudaNode*)in.nodes)[getLo(leaves[lidx])];
		int idx0 = node->idx0;
		int idx1 = node->idx1;
		bool found = false; // any points inside this node?

		// check if any point inside node
		for (; idx0 < idx1; idx0++)
		{
			// Reproject to output sample's (u,v,t)
			float4 data0 = tex1Dfetch(t_points, idx0*3+0);
			float4 data1 = tex1Dfetch(t_points, idx0*3+1);
			float ptx   = data1.x;
			float pty   = data1.y;
			float ptw   = data0.x;
			float ptt   = data0.y;
			float ptmvx = data1.z;
			float ptmvy = data1.w;
			float ptmvw = data0.z;

			float p0x   = ptx * ptw;
			float p0y   = pty * ptw;
			float p0w   = ptw;
			float px    = p0x + (st - ptt) * ptmvx;
			float py    = p0y + (st - ptt) * ptmvy;
			float pw    = p0w + (st - ptt) * ptmvw;
			float oow   = 1.f/pw;
			float cocr  = in.cocCoeffs.x * oow + in.cocCoeffs.y;
			float x     = px * oow + cocr * su;
			float y     = py * oow + cocr * sv;
			float dx    = x - sx;
			float dy    = y - sy;
			float dist2 = dx*dx + dy*dy;

			// inside radius?
			if (dist2 < radius*radius)
			{
				found = true;
				break; // won't need the nearest
			}
		}

		if (!found)
			empty |= (((U64)1)<<lidx); // ignore this leaf
		else
		{
			// test previous leaves of the currently open surface
			for (int j = 0; j < lidx; ++j)
			{
				float RANGE = 1.0f/sqrtf(128.0f);
				Vec3f uvtmin = Vec3f(su-RANGE, sv-RANGE, st - 0.5f*RANGE);
				Vec3f uvtmax = Vec3f(su+RANGE, sv+RANGE, st + 0.5f*RANGE);
				const CudaNode* prev = &((const CudaNode*)in.nodes)[getLo(leaves[j])];
				if (!(empty & (((U64)1)<<j)))
				{
					if (!sameSurface(*node, *prev, uvtmin, uvtmax))
					{
						multi = true;
						break; // break as soon as conflict is found
					}
				}
			}
		}

		// break if multiple surfaces found
		if (multi)
			break;

		// process the remaining points
		for (; idx0 < idx1; idx0++)
		{
			// Reproject to output sample's (u,v,t)
			float4 data0 = tex1Dfetch(t_points, idx0*3+0);
			float4 data1 = tex1Dfetch(t_points, idx0*3+1);
			float ptx   = data1.x;
			float pty   = data1.y;
			float ptw   = data0.x;
			float ptt   = data0.y;
			float ptmvx = data1.z;
			float ptmvy = data1.w;
			float ptmvw = data0.z;

			float p0x   = ptx * ptw;
			float p0y   = pty * ptw;
			float p0w   = ptw;
			float px    = p0x + (st - ptt) * ptmvx;
			float py    = p0y + (st - ptt) * ptmvy;
			float pw    = p0w + (st - ptt) * ptmvw;
			float oow   = 1.f/pw;
			float cocr  = in.cocCoeffs.x * oow + in.cocCoeffs.y;
			float x     = px * oow + cocr * su;
			float y     = py * oow + cocr * sv;
			float dx    = x - sx;
			float dy    = y - sy;
			float dist2 = dx*dx + dy*dy;

			// inside radius?
			if (dist2 <= radius*radius)
			{
				float4 data2 = tex1Dfetch(t_points, idx0*3+2);

				float k = dx/dy;
				if (dy >= 0.f)	wbox[0] = fmin(wbox[0], k),	wbox[1] = fmax(wbox[1], k);
				else			wbox[2] = fmin(wbox[2], k),	wbox[3] = fmax(wbox[3], k);

				// color
				float w = fmax(0.f, 1.f - sqrtf(dist2) / radius);
				color += w * Vec4f(data2.x, data2.y, data2.z, data2.w);
			}
		}
	}

	// we may have certain coverage from the first surface
	if (color.w > 0.f && wbox[1] > wbox[2] && wbox[0] < wbox[3])
	{
		color *= 1.f / color.w;
		storeResult(tidx, color);
		return;
	}

	// multiple surfaces -> go 2R
	if (multi)
	{
		addFailed(tidx);
		return;
	}

	// get nearest within 2R
	addNearest(tidx);
}

//------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(128,7) nearestKernel(void)
{
	__shared__ volatile int s_foo[128];
    const RecKernelInput& in = *(const RecKernelInput*)&c_RecKernelInput;

    int tidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (tidx >= g_nearestCount)
		return;

	U32 sampleIdx = ((U32*)in.nearest)[tidx];

	CudaSample s;
	constructOutputSample(sampleIdx, s);

	float su = s.u;
	float sv = s.v;
	float st = s.t;
	float sx = s.x;
	float sy = s.y;

	// find the nearest x,y point
	enum { STACK_SIZE = 32 };
	U32 stack[STACK_SIZE];
	S32 stackPtr;
	int minIndex = -1;

	// gradually enlarge search radius until something is found
	for (float rmul = 1.f; rmul <= 16.f; rmul *= 2.f)
	{
		float radius = rmul * in.dispersion;
		stackPtr = 0;
		stack[0] = in.rootNode;
		while(stackPtr >= 0)
		{
			int idx = stack[stackPtr];
			stackPtr--;

			float4 planes0 = tex1Dfetch(t_nodes, idx*4+0);
			float4 planes1 = tex1Dfetch(t_nodes, idx*4+1);
			float4 planes2 = tex1Dfetch(t_nodes, idx*4+2);
			float4 coc_idx = tex1Dfetch(t_nodes, idx*4+3);
			if (intersectNodeSample2(planes0, planes1, planes2, su, sv, st, sx, sy, radius))
			{
				int idx0 = __float_as_int(coc_idx.z);
				int idx1 = __float_as_int(coc_idx.w);
				if (idx0 <= idx1)
				{
					for (; idx0 < idx1; idx0++)
					{
						// Reproject to output sample's (u,v,t)
						float4 data0 = tex1Dfetch(t_points, idx0*3+0);
						float4 data1 = tex1Dfetch(t_points, idx0*3+1);
						float ptx   = data1.x;
						float pty   = data1.y;
						float ptw   = data0.x;
						float ptt   = data0.y;
						float ptmvx = data1.z;
						float ptmvy = data1.w;
						float ptmvw = data0.z;

						float p0x   = ptx * ptw;
						float p0y   = pty * ptw;
						float p0w   = ptw;
						float px    = p0x + (st - ptt) * ptmvx;
						float py    = p0y + (st - ptt) * ptmvy;
						float pw    = p0w + (st - ptt) * ptmvw;
						float oow   = 1.f/pw;
						float cocr  = in.cocCoeffs.x * oow + in.cocCoeffs.y;
						float x     = px * oow + cocr * su;
						float y     = py * oow + cocr * sv;
						float dx    = x - sx;
						float dy    = y - sy;
						float dist2 = dx*dx + dy*dy;

						// update nearest and search radius
						if (dist2 <= radius*radius)
						{
							radius   = sqrtf(dist2);
							minIndex = idx0;
						}
					}
				}
				else
				{
					stackPtr += 2;
					stack[stackPtr-1] = idx0;
					stack[stackPtr  ] = idx1;
				}
			}
		}

		// if something found, it is the nearest -> write it and exit
		if (minIndex >= 0)
		{
			float4 data2 = tex1Dfetch(t_points, minIndex * 3 + 2);
			storeResult(sampleIdx, Vec4f(data2.x, data2.y, data2.z, data2.w));
			return;
		}
	}

	// bummer
}

//------------------------------------------------------------------------

__device__ inline float lenSqr(const Vec2f& v)
{
	return v.x*v.x + v.y*v.y;
}

__device__ bool checkTriangles(const U64* xy, int numPoints, float dispersion)
{
	if (numPoints < 3)
		return false;

	enum { N = 128 }; // must be equal to number of points
	U64 goodxy[N];	  // good dudes coordinates

	float dcomp = 4.f*dispersion*dispersion;
	for (int i=0; i < numPoints-2; i++)
	{
		Vec2f p0 = U64toVec2f(xy[i]);

		int pgood = 0; // positive signs
		int ngood = N; // negative signs
		for (int j=i+1; j < numPoints; j++)
		{
			Vec2f p1 = U64toVec2f(xy[j]);
			if (lenSqr(p0-p1) < dcomp)
			{
				float c0 = p0.x*p1.y - p1.x*p0.y;
				int idx;
				if (c0 >= 0.f)
					idx = pgood++;
				else
					idx = --ngood;
				goodxy[idx] = Vec2ftoU64(p1);
			}
		}
		
		if (ngood != N)
		for (int j=0; j < pgood; j++) // loop over positive
		{
			Vec2f p1 = U64toVec2f(goodxy[j]);
			for(int k=ngood; k < N; k++) // loop over negative
			{
				Vec2f p2 = U64toVec2f(goodxy[k]);
				if (p1.x*p2.y >= p2.x*p1.y && lenSqr(p1-p2) < dcomp)
					return true;
			}
		}
	}

	return false;
}

//------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(128,7) failKernel(void)
{
	__shared__ volatile int s_foo[128];
    const RecKernelInput& in = *(const RecKernelInput*)&c_RecKernelInput;

    int tidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (tidx >= g_failedCount)
		return;

	U32 sampleIdx = ((U32*)in.failed)[tidx];
	CudaSample s;
	constructOutputSample(sampleIdx, s);
	float dispersion = in.dispersion;	
	float radius     = dispersion * 2.f;

	// it's business time!

	enum { STACK_SIZE = 32 };
	U32 stack[STACK_SIZE];
	U64 leaves[MAX_LEAVES_R2];	// lo = idx, hi = key
	S32 stackPtr = 0;
	int numLeaves = 0;
	stack[0] = in.rootNode;

	float su = s.u;
	float sv = s.v;
	float st = s.t;
	float sx = s.x;
	float sy = s.y;
	while(stackPtr >= 0)
	{
		int idx = stack[stackPtr];
		stackPtr--;

		float4 planes0 = tex1Dfetch(t_nodes, idx*4+0);
		float4 planes1 = tex1Dfetch(t_nodes, idx*4+1);
		float4 planes2 = tex1Dfetch(t_nodes, idx*4+2);
		float4 coc_idx = tex1Dfetch(t_nodes, idx*4+3);
		if (intersectNodeSample2(planes0, planes1, planes2, su, sv, st, sx, sy, radius))
		{
			int idx0 = __float_as_int(coc_idx.z);
			int idx1 = __float_as_int(coc_idx.w);
			if (idx0 <= idx1)
			{
				if (numLeaves == MAX_LEAVES_R2)
				{
					numLeaves++;
					break;
				}

				const CudaPoint& p0 = ((const CudaPoint*)in.points)[idx0];
				float w = p0.w + (st - p0.t) * p0.mvw;
				leaves[numLeaves] = combineLoHi(idx, __float_as_int(w));
				numLeaves++;
			}
			else
			{
				stackPtr += 2;
				stack[stackPtr-1] = idx0;
				stack[stackPtr  ] = idx1;
			}
		}
	}

	// no leaves, just find nearest sample and hope for the best
	if (!numLeaves)
	{
		addNearest(sampleIdx);
		return;
	}

	if (numLeaves > MAX_LEAVES_R2)
	{
		// insert overflow entry and bail out
		addOverflow(sampleIdx);
		return;
	}

	// sort leaves
	for (int i=0; i < numLeaves-1; i++)
	{
		float w  = __int_as_float(getHi(leaves[i]));
		int   mn = i;
		for (int j=i+1; j < numLeaves; j++)
		{
			float w1 = __int_as_float(getHi(leaves[j]));
			if (w1 < w)
			{
				w  = w1;
				mn = j;
			}
		}
		if (mn != i)
			swap(leaves[i], leaves[mn]);
	}

	// reconstruct one surface at a time
	U64   xy[MAX_POINTS];	// projected sample x for triangle test
	int   numPoints;		// points in current surface
	int   sfirst;			// first of surface
	Vec4f color; 			// accumulate color of current surface
	enum { MIN_SAMPLES_PER_SURFACE = 4 };	// Arbitrary (4 means four quadrants)
	int   lidx = 0;
	Vec4f abox;
	Vec4f wbox;
	do
	{
		// start collecting a surface here
		sfirst    = lidx;
		numPoints = 0;
		color.x   = 0.f;
		color.y   = 0.f;
		color.z   = 0.f;
		color.w   = 0.f;
		abox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
		wbox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
		do
		{
			// do we end the currently open surface here?
			const CudaNode* node = &((const CudaNode*)in.nodes)[getLo(leaves[lidx])];
			bool endSurface = false;

			// don't consider ending if there's too few samples
			if (numPoints >= MIN_SAMPLES_PER_SURFACE)
			{
				for (int j = sfirst; j < lidx; ++j)
				{
					float RANGE = 1.0f/sqrtf(128.0f);
					Vec3f uvtmin = Vec3f(su-RANGE, sv-RANGE, st - 0.5f*RANGE);
					Vec3f uvtmax = Vec3f(su+RANGE, sv+RANGE, st + 0.5f*RANGE);
					const CudaNode* prev = &((const CudaNode*)in.nodes)[getLo(leaves[j])];
					if (!sameSurface(*node, *prev, uvtmin, uvtmax))
					{
						endSurface = true;
						break; // break as soon as conflict is found
					}
				}
			}

			// this surface is done, go test it
			if (endSurface)
				break;

			// add samples to the current surface
			int idx0 = node->idx0;
			int idx1 = node->idx1;
			for (int i=idx0; i < idx1; i++)
			{
				// Reproject to output sample's (u,v,t)
				CudaPoint p;
				float4 data0 = tex1Dfetch(t_points, i*3+0);
				float4 data1 = tex1Dfetch(t_points, i*3+1);
				float ptx   = data1.x;
				float pty   = data1.y;
				float ptw   = data0.x;
				float ptt   = data0.y;
				float ptmvx = data1.z;
				float ptmvy = data1.w;
				float ptmvw = data0.z;

				float p0x   = ptx * ptw;
				float p0y   = pty * ptw;
				float p0w   = ptw;
				float pnx   = p0x + (st - ptt) * ptmvx;
				float pny   = p0y + (st - ptt) * ptmvy;
				float pnw   = p0w + (st - ptt) * ptmvw;
				float oow   = 1.f/pnw;
				float cocr  = in.cocCoeffs.x * oow + in.cocCoeffs.y;
				float x     = pnx * oow + cocr * su;
				float y     = pny * oow + cocr * sv;
				float dx    = x - sx;
				float dy    = y - sy;
				float dist2 = dx*dx + dy*dy;

				// inside radius?
				if (dist2 <= radius*radius)
				{
					if (numPoints >= MAX_POINTS)
					{
						// we're toasted, overflow and exit
						addOverflow(sampleIdx);
						return;
					}

					xy[numPoints] = combineLoHi(__float_as_int(dx), __float_as_int(dy));
					numPoints++;	

					float w = 1.f - sqrtf(dist2) / dispersion;		// tent filter

					float k = dx/dy;
					if (w <= 0.f)
					{
						if (dy >= 0.f)	abox[0] = fmin(abox[0], k), abox[1] = fmax(abox[1], k);
						else			abox[2] = fmin(abox[2], k), abox[3] = fmax(abox[3], k);
					}
					else
					{
						if (dy >= 0.f)	wbox[0] = fmin(wbox[0], k),	wbox[1] = fmax(wbox[1], k);
						else			wbox[2] = fmin(wbox[2], k),	wbox[3] = fmax(wbox[3], k);

						float4 data2 = tex1Dfetch(t_points, i*3+2);
						color += w * Vec4f(data2.x, data2.y, data2.z, data2.w);						
					}
				}
			} 

			// ok this leaf is now added to the surface, try the next one
			lidx++;
		} while (lidx < numLeaves);
		
		// end of surface here

		// test triangles, but only if not the last surface (out of leaves)
		if (lidx < numLeaves)
		{
			abox[0] = fmin(abox[0], wbox[0]);
			abox[1] = fmax(abox[1], wbox[1]);
			abox[2] = fmin(abox[2], wbox[2]);
			abox[3] = fmax(abox[3], wbox[3]);
			if (abox[1] > abox[2] && abox[0] < abox[3])
			{
				if ((wbox[1] > wbox[2] && wbox[0] < wbox[3]) || checkTriangles(xy, numPoints, dispersion))
					break;
			}
		}

		// end if out of leaves
	} while(lidx < numLeaves);

	// all done, either we have a color or we resort to nearest find
	if (color.w > 0.f)
		storeResult(sampleIdx, color * (1.f / color.w));
	else
		addNearest(sampleIdx);
}
