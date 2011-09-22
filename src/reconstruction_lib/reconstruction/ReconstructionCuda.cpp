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
// CUDA reconstruction
// 
// To use the GPU version, you MUST open src/framework/base/DLLImports.hpp, and change the line
// 
// #define FW_USE_CUDA 0
// 
// to
// 
// #define FW_USE_CUDA 1.
// 
// In addition, you must have NVIDIA Cuda Toolkit 4.0 properly installed.
//-------------------------------------------------------------------------------------------------

#pragma warning(disable:4127)		// conditional expression is constant
#include "Reconstruction.hpp"
#include "gpu/CudaCompiler.hpp"
#include "ReconstructionCudaKernels.hpp"
#include <cfloat>
#include <cstdio>

using namespace FW;

//-------------------------------------------------------------------------------------------------

class CudaReconstruction
{
public:
							CudaReconstruction		(void);
							~CudaReconstruction		(void);

	void					init					(int spp, int outputSpp, int numPatterns, const Vec2i& size,
													 Image& resultImage,
													 const Array<CudaSample>& outputSamples,
													 Array<Vec4f>& results,
													 Array<CudaNode>& nodes,
													 int rootNode,
													 Array<CudaTraversalNode>& tnodes,
													 Array<CudaPoint>& points,
													 const Vec2f& cocCoeffs);
	void					reconstructCPU			(void);
	void					reconstructGPU			(void);

	static void				reconstructCPUBatchDisp	(MulticoreLauncher::Task& task);
	void					reconstructCPUBatch		(U32* overflow, int first, int idx0, int idx1);

	struct CPUBatchTask
	{
		CudaReconstruction*	obj;
		U32*				overflow;
		int					first;
		int					idx0;
		int					idx1;
	};

private:
	void					constructOutputSample	(U32 id, CudaSample& s);
	float					getDispersion			(void);
	Vec4f					reconstructCPUSingle	(int smp);
	Vec4f					reconstructCPUNearest	(int smp);
	Vec4f					reconstructCPUFailed	(int smp);
	int						collectNodes			(Array<int>& leaves, const CudaSample& s, float radius);
	bool					intersectNodeSample		(const CudaNode& node, const CudaSample& s, float radius);

    CudaCompiler			m_compiler;

	int							m_spp;
	int							m_outputSpp;
	int							m_numPatterns;
	Vec2i						m_size;
	Image*						m_resultImage;
	const Array<CudaSample>*	m_outputSamples;	// the "templates"
	Array<Vec4f>*				m_results;
	Array<CudaNode>*			m_nodes;
	Array<CudaTraversalNode>*	m_traversalNodes;
	Array<CudaPoint>*			m_points;
	int							m_rootNode;
	Vec2f						m_cocCoeffs;
};

CudaReconstruction::CudaReconstruction(void)
{
    m_compiler.setSourceFile("src/reconstruction_lib/reconstruction/ReconstructionCudaKernels.cu");
    m_compiler.include("src/framework");
    m_compiler.define("SM_ARCH", sprintf("%d", CudaModule::getComputeCapability()));
    m_compiler.define("FW_USE_CUDA", "1");
}

CudaReconstruction::~CudaReconstruction(void)
{
	// empty
}

//-------------------------------------------------------------------------------------------------

static inline int ternaryCompare( float a, float b, float eps )
{
	if ( fabs(a-b) < eps )
		return 0;
	else if ( a < b )
		return -1;
	else
		return 1;
}

static inline bool ternaryEqual( int a, int b )
{
	return a==0 || b==0 || a==b;
}

//-------------------------------------------------------------------------------------------------

static bool sameSurface(const CudaNode& n0,const CudaNode& n1, const Vec3f& uvtmin, const Vec3f& uvtmax)
{
	const float COC_THRESHOLD = 1*0.1f;

	// evaluate U bounds at t between a and b
	float tmid = 0.5f * (uvtmin.z+uvtmax.z);

	const int XMIN=0;
	const int XMAX=1;
	const int YMIN=2;
	const int YMAX=3;

	float Xmin00 = dot( n0.planes[XMIN], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmin10 = dot( n0.planes[XMIN], Vec3f( tmid, uvtmax.x, 1.0f ) );
	float Xmax00 = dot( n0.planes[XMAX], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmax10 = dot( n0.planes[XMAX], Vec3f( tmid, uvtmax.x, 1.0f ) );

	float Ymin00 = dot( n0.planes[YMIN], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymin10 = dot( n0.planes[YMIN], Vec3f( tmid, uvtmax.y, 1.0f ) );
	float Ymax00 = dot( n0.planes[YMAX], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymax10 = dot( n0.planes[YMAX], Vec3f( tmid, uvtmax.y, 1.0f ) );

	float Xmin01 = dot( n1.planes[XMIN], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmin11 = dot( n1.planes[XMIN], Vec3f( tmid, uvtmax.x, 1.0f ) );
	float Xmax01 = dot( n1.planes[XMAX], Vec3f( tmid, uvtmin.x, 1.0f ) );
	float Xmax11 = dot( n1.planes[XMAX], Vec3f( tmid, uvtmax.x, 1.0f ) );

	float Ymin01 = dot( n1.planes[YMIN], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymin11 = dot( n1.planes[YMIN], Vec3f( tmid, uvtmax.y, 1.0f ) );
	float Ymax01 = dot( n1.planes[YMAX], Vec3f( tmid, uvtmin.y, 1.0f ) );
	float Ymax11 = dot( n1.planes[YMAX], Vec3f( tmid, uvtmax.y, 1.0f ) );

	bool bCoC =
		   ternaryEqual( ternaryCompare( Xmin00, Xmin01, COC_THRESHOLD ), ternaryCompare( Xmin10, Xmin11, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( Ymin00, Ymin01, COC_THRESHOLD ), ternaryCompare( Ymin10, Ymin11, COC_THRESHOLD ) ) && 
		   ternaryEqual( ternaryCompare( Xmax00, Xmax01, COC_THRESHOLD ), ternaryCompare( Xmax10, Xmax11, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( Ymax00, Ymax01, COC_THRESHOLD ), ternaryCompare( Ymax10, Ymax11, COC_THRESHOLD ) );

	if ( !bCoC )
		return false;
						 
	// evaluate XY (without UV) bounds at ta,tb

	float xmin_ta_node0 = dot( n0.planes[XMIN], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float xmin_tb_node0 = dot( n0.planes[XMIN], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float ymin_ta_node0 = dot( n0.planes[YMIN], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float ymin_tb_node0 = dot( n0.planes[YMIN], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float xmax_ta_node0 = dot( n0.planes[XMAX], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float xmax_tb_node0 = dot( n0.planes[XMAX], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float ymax_ta_node0 = dot( n0.planes[YMAX], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float ymax_tb_node0 = dot( n0.planes[YMAX], Vec3f( uvtmax.z, 0.0f, 1.0f ) );

	float xmin_ta_node1 = dot( n1.planes[XMIN], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float xmin_tb_node1 = dot( n1.planes[XMIN], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float ymin_ta_node1 = dot( n1.planes[YMIN], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float ymin_tb_node1 = dot( n1.planes[YMIN], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float xmax_ta_node1 = dot( n1.planes[XMAX], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float xmax_tb_node1 = dot( n1.planes[XMAX], Vec3f( uvtmax.z, 0.0f, 1.0f ) );
	float ymax_ta_node1 = dot( n1.planes[YMAX], Vec3f( uvtmin.z, 0.0f, 1.0f ) );
	float ymax_tb_node1 = dot( n1.planes[YMAX], Vec3f( uvtmax.z, 0.0f, 1.0f ) );

	bool bTime =
		   ternaryEqual( ternaryCompare( xmin_ta_node0, xmin_ta_node1, COC_THRESHOLD ), ternaryCompare( xmin_tb_node0, xmin_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( xmax_ta_node0, xmax_ta_node1, COC_THRESHOLD ), ternaryCompare( xmax_tb_node0, xmax_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( ymin_ta_node0, ymin_ta_node1, COC_THRESHOLD ), ternaryCompare( ymin_tb_node0, ymin_tb_node1, COC_THRESHOLD ) ) &&
		   ternaryEqual( ternaryCompare( ymax_ta_node0, ymax_ta_node1, COC_THRESHOLD ), ternaryCompare( ymax_tb_node0, ymax_tb_node1, COC_THRESHOLD ) );

	return bTime;
}

//-------------------------------------------------------------------------------------------------

static bool checkTriangles(const Array<Vec2f>& xy, float dispersion)
{
	for(int m=0  ; m < xy.getSize()-2; m++)
	for(int k=m+1; k < xy.getSize()-1; k++)
	for(int l=k+1; l < xy.getSize()  ; l++)
	{
		Vec2f p0 = xy[m];
		Vec2f p1 = xy[k];
		Vec2f p2 = xy[l];

		// check that edge lengths are short enough
		if ((p0-p1).length() > 2*dispersion ||
			(p1-p2).length() > 2*dispersion ||
			(p2-p0).length() > 2*dispersion)
			continue;

		// check coverage
		float c0 = p0.x*p1.y - p1.x*p0.y;
		float c1 = p1.x*p2.y - p2.x*p1.y;
		float c2 = p2.x*p0.y - p0.x*p2.y;
		if (c0 <= 0.f && c1 <= 0.f && c2 <= 0.f) return true;
		if (c0 >= 0.f && c1 >= 0.f && c2 >= 0.f) return true;
	}
	return false;
}

//-------------------------------------------------------------------------------------------------

void CudaReconstruction::init(int spp, int outputSpp, int numPatterns, const Vec2i& size, 
							  Image& resultImage,
							  const Array<CudaSample>& outputSamples,
							  Array<Vec4f>& results, Array<CudaNode>& nodes, int rootNode, 
							  Array<CudaTraversalNode>& tnodes, Array<CudaPoint>& points,
							  const Vec2f& cocCoeffs)
{
	m_spp            = spp;
	m_outputSpp      = outputSpp;
	m_numPatterns    = numPatterns;
	m_size           = size;
	m_resultImage    = &resultImage;
	m_outputSamples  = &outputSamples;
	m_results        = &results;
	m_nodes          = &nodes;
	m_rootNode       = rootNode;
	m_traversalNodes = &tnodes;
	m_points         = &points;
	m_cocCoeffs      = cocCoeffs;
}

void CudaReconstruction::reconstructCPU(void)
{
	// clear result image
	for (int y=0; y < m_size.y; y++)
	for (int x=0; x < m_size.x; x++)
		m_resultImage->setVec4f(Vec2i(x, y), Vec4f(0));

	// process output samples
	U32 N = m_size.x * m_size.y * m_outputSpp;
	float prev = -1.f;
	FW::printf("N=%u\n", N);
	for (U32 i=0; i < N; i++)
	{
		float prog = floorf(1000.f * (float)i / (float)N);
		if (prog > prev)
		{
			prev = prog;
			FW::printf("%5.1f %% ... \r", 0.1f*prog);
		}
		fflush(stdout);
		Vec4f c = reconstructCPUSingle(i);

		// add into result
		U32 y = i / (m_size.x * m_outputSpp);
		U32 x = (i - y * m_size.x * m_outputSpp) / m_outputSpp;
		m_resultImage->setVec4f(Vec2i(x, y), m_resultImage->getVec4f(Vec2i(x, y)) + c);
	}

	// normalize result image
	for (int y=0; y < m_size.y; y++)
	for (int x=0; x < m_size.x; x++)
	{
		Vec4f c = m_resultImage->getVec4f(Vec2i(x, y));
		if (c.w == 0.f)
			c = Vec4f(1,0,0,1);
		else
			c /= c.w;
		m_resultImage->setVec4f(Vec2i(x, y), c);
	}

	FW::printf("100.0 %%       \n");
}

//-------------------------------------------------------------------------------------------------

void CudaReconstruction::reconstructCPUBatchDisp(MulticoreLauncher::Task& task)
{
	CPUBatchTask* bt = (CPUBatchTask*)task.data;
	bt->obj->reconstructCPUBatch(bt->overflow, bt->first, bt->idx0, bt->idx1);
}

void CudaReconstruction::reconstructCPUBatch(U32* overflow, int first, int idx0, int idx1)
{
	for (int i=idx0; i < idx1; i++)
	{
		int idx = overflow[i];
		Vec4f c = reconstructCPUSingle(idx + first);
		U32 id = idx + first;
		U32 y = id / (m_size.x * m_outputSpp);
		U32 x = (id - y * m_size.x * m_outputSpp) / m_outputSpp;
		m_resultImage->setVec4f(Vec2i(x, y), m_resultImage->getVec4f(Vec2i(x, y)) + c);
	}
}

//-------------------------------------------------------------------------------------------------

void CudaReconstruction::reconstructGPU(void)
{
#if !FW_USE_CUDA
	fail("Not built with FW_USE_CUDA!" );
#endif

	// set up kernel
	CudaModule* module = m_compiler.compile();
	failIfError();
	CUfunction kernel        = module->getKernel("recKernel");
	CUfunction nearestKernel = module->getKernel("nearestKernel");
	CUfunction failKernel    = module->getKernel("failKernel");
	if (!kernel || !failKernel)
		fail("CUDA kernels not found!");

	// clear CPU result image
	for (int y=0; y < m_size.y; y++)
	for (int x=0; x < m_size.x; x++)
		m_resultImage->setVec4f(Vec2i(x, y), Vec4f(0));

	// TODO make these live in buffers
	Buffer osmp   (m_outputSamples->getPtr(), m_outputSamples->getNumBytes());
	Buffer nodes  (m_nodes->getPtr(), m_nodes->getNumBytes());
	Buffer tnodes (m_traversalNodes->getPtr(), m_traversalNodes->getNumBytes());
	Buffer points (m_points->getPtr(), m_points->getNumBytes());
#if FW_USE_CUDA
	module->setTexRef("t_nodes",  tnodes, CU_AD_FORMAT_FLOAT, 4);
	module->setTexRef("t_points", points, CU_AD_FORMAT_FLOAT, 4);
#endif

	U32 numSamples = m_size.x * m_size.y * m_outputSpp;
	U32 maxBatchSize = 1024*1024*2;
	U32 numBatches = (numSamples+maxBatchSize-1)/maxBatchSize;
	FW::printf("batches: %d\n", numBatches);
	float t = 0.f;

	Buffer nearest  (0, maxBatchSize * sizeof(U32));
	Buffer failed   (0, maxBatchSize * sizeof(U32));
	Buffer overflow (0, maxBatchSize * sizeof(U32));
	Buffer resultImg(0, m_size.x * m_size.y * sizeof(Vec4f));
	resultImg.clear(); // GPU result image

	RecKernelInput& in = *(RecKernelInput*)module->getGlobal("c_RecKernelInput").getMutablePtr();
	in.osmp        = osmp.getCudaPtr();
	in.nodes       = nodes.getCudaPtr();
	in.points      = points.getCudaPtr();
	in.nearest     = nearest.getMutableCudaPtr();
	in.failed      = failed.getMutableCudaPtr();
	in.resultImg   = resultImg.getMutableCudaPtr();
	in.overflow    = overflow.getMutableCudaPtr();
	in.cocCoeffs   = m_cocCoeffs;
	in.dispersion  = getDispersion();
	in.rootNode    = m_rootNode;
	in.spp         = (float)m_spp;
#if FW_USE_CUDA
	in.size        = make_int2(m_size.x, m_size.y);
#endif
	in.outputSpp   = m_outputSpp;
	in.numPatterns = m_numPatterns;

	int failedTotal   = 0;
	int overflowTotal = 0;
	Timer timer;
	timer.start();
	for (U32 batch = 0; batch < numBatches; batch++)
	{
		FW::printf("%d / %d ..\r", batch, numBatches);
		U32 first = batch * maxBatchSize;
		U32 num   = min(numSamples - first, maxBatchSize);

		*((U32*)module->getGlobal("g_failedCount")  .getMutablePtr()) = 0;
		*((U32*)module->getGlobal("g_nearestCount") .getMutablePtr()) = 0;
		*((U32*)module->getGlobal("g_overflowCount").getMutablePtr()) = 0;

		RecKernelInput& in = *(RecKernelInput*)module->getGlobal("c_RecKernelInput").getMutablePtr();
		in.firstSample = first;
		in.numSamples  = num;
		module->updateGlobals();

		Vec2i blockSize(32, 4);
		Vec2i gridSize((num + blockSize.x * blockSize.y - 1) / (blockSize.x * blockSize.y), 1);
		t += module->launchKernelTimed(kernel,        blockSize, gridSize);
		t += module->launchKernelTimed(failKernel,    blockSize, gridSize);
		t += module->launchKernelTimed(nearestKernel, blockSize, gridSize);

		// process the overflow samples on CPU
		int numOverflow = *((U32*)module->getGlobal("g_overflowCount").getPtr());
		if (numOverflow)
		{
			U32* optr = (U32*)overflow.getPtr();
			MulticoreLauncher launcher;
			int numTasks = 32;
			Array<CPUBatchTask> bt(0, numTasks);
			int bsize = (numOverflow + numTasks - 1) / numTasks;
			for(int i=0; i < numTasks; i++)
			{
				bt[i].obj      = this;
				bt[i].overflow = optr;
				bt[i].first    = first;
				bt[i].idx0     = bsize * i;
				bt[i].idx1     = min(bsize * (i+1), numOverflow);
				if (bt[i].idx0 < bt[i].idx1)
					launcher.push(reconstructCPUBatchDisp, &bt[i]);
			}
			launcher.popAll();
			in.overflow = overflow.getMutableCudaPtr(); // make sure GPU updates this
		}

		failedTotal   += *((U32*)module->getGlobal("g_failedCount").getPtr());
		overflowTotal += numOverflow;
	}

	// add GPU result image to CPU result image
	Vec4f* rptr = (Vec4f*)resultImg.getPtr();
	for (int y=0; y < m_size.y; y++)
	for (int x=0; x < m_size.x; x++)
		m_resultImage->setVec4f(Vec2i(x, y), m_resultImage->getVec4f(Vec2i(x, y)) + rptr[x+m_size.x*y]);

	// normalize image
	for (int y=0; y < m_size.y; y++)
	for (int x=0; x < m_size.x; x++)
	{
		Vec4f c = m_resultImage->getVec4f(Vec2i(x, y));
		if (c.w == 0.f)
			c = Vec4f(1,0,0,1);
		else
			c *= (1.f/c.w);
		m_resultImage->setVec4f(Vec2i(x, y), c);
	}

	timer.end();

	// stats
	FW::printf("\n");
	FW::printf("kernel launch time: %.3f s\n", t);
	FW::printf("total CPU+GPU time: %.3f s\n", timer.getTotal());
	FW::printf("non-R1 samples:   %d = %.2f %%\n", failedTotal, 100.f * failedTotal / numSamples);
	FW::printf("overflow samples: %d = %.2f %%\n", overflowTotal, 100.f * overflowTotal / numSamples);
}

//-------------------------------------------------------------------------------------------------

void CudaReconstruction::constructOutputSample(U32 id, CudaSample& s)
{
	U32 y = id / (m_size.x * m_outputSpp);
	id -= y * m_size.x * m_outputSpp;
	U32 x = id / m_outputSpp;
	id -= x * m_outputSpp;
	U32 pattern = hashBits(x, y) % m_numPatterns;
	const CudaSample& is = (*m_outputSamples)[pattern * m_outputSpp+id];
	s.x = is.x + (float)x;
	s.y = is.y + (float)y;
	s.u = is.u;
	s.v = is.v;
	s.t = is.t;
}

float CudaReconstruction::getDispersion(void)
{
	switch(m_spp)
	{
	default:	fail("CudaReconstruction::getDispersion -- unknown SPP");
	case 256:	return 0.14f;
	case 128:	return 0.18f;
	case 64:	return 0.27f;
	case 32:	return 0.37f;
	case 16:	return 0.50f;
	case 8:		return 0.80f;
	case 4:		return 1.10f;
	case 2:		return 1.34f;
	case 1:		return 2.00f;
	}
}

bool CudaReconstruction::intersectNodeSample(const CudaNode& node, const CudaSample& s, float radius)
{
	const int XMIN=0;
	const int XMAX=1;
	const int YMIN=2;
	const int YMAX=3;

	// evaluate four hyperplanes to get a conservative XY bounding rect for node at sample's t,u,v, dilate by R
	Vec2f mn = Vec2f( dot( node.planes[XMIN], Vec3f(s.t,s.u,1.0f) ), dot( node.planes[YMIN], Vec3f(s.t,s.v,1.0f) ) ) - radius;
	Vec2f mx = Vec2f( dot( node.planes[XMAX], Vec3f(s.t,s.u,1.0f) ), dot( node.planes[YMAX], Vec3f(s.t,s.v,1.0f) ) ) + radius;

	// check if sample's xy lies within rect
	return (s.x>=mn.x && s.y>=mn.y && s.x<=mx.x && s.y<=mx.y);
}

int CudaReconstruction::collectNodes(Array<int>& leaves, const CudaSample& s, float radius)
{
	leaves.clear();

	Array<int> stack;
	stack.add(m_rootNode);

	while(stack.getSize())
	{
		int idx = stack.removeLast();
		const CudaNode& node = (*m_nodes)[idx];

		if (intersectNodeSample(node, s, radius))
		{
			if (node.isLeaf())
				leaves.add(idx);
			else
			{
				stack.add(node.idx0);
				stack.add(node.idx1);
			}
		}
	}

	// nada?
	if (!leaves.getSize())
		return 0;

	// sort leaves
	for (int i=0; i < leaves.getSize()-1; i++)
	{
		const CudaNode& n0 = (*m_nodes)[leaves[i]];
		const CudaPoint& p0 = (*m_points)[n0.idx0];
		float w  = p0.w + (s.t - p0.t) * p0.mvw;
		int   mn = i;
		for (int j=i+1; j < leaves.getSize(); j++)
		{
			const CudaNode& n1 = (*m_nodes)[leaves[j]];
			const CudaPoint& p1 = (*m_points)[n1.idx0];
			float w1 = p1.w + (s.t - p1.t) * p1.mvw;
			if (w1 < w)
			{
				w  = w1;
				mn = j;
			}
		}
		if (mn != i)
			swap(leaves[i], leaves[mn]);
	}

	return leaves.getSize();
}

//-------------------------------------------------------------------------------------------------

Vec4f CudaReconstruction::reconstructCPUSingle(int id)
{
	CudaSample s;
	constructOutputSample(id, s);
	float dispersion = getDispersion();
	float radius     = dispersion;

	// 1R fetch
	Array<int> leaves;
	collectNodes(leaves, s, radius);

	// remove leaves that have no points inside
	int lcount = 0;
	for(int i=0; i < leaves.getSize(); i++)
  	{
  		const CudaNode& node = (*m_nodes)[leaves[i]];
  
		bool empty = true;
  		for(int j=node.idx0; j < node.idx1; j++)
  		{
  			// Reproject to output sample's (u,v,t)
  			const CudaPoint& p = (*m_points)[j];
  			float p0x   = p.x * p.w;
  			float p0y   = p.y * p.w;
  			float p0w   = p.w;
  			float px    = p0x + (s.t - p.t) * p.mvx;
  			float py    = p0y + (s.t - p.t) * p.mvy;
  			float pw    = p0w + (s.t - p.t) * p.mvw;
  			float cocr  = m_cocCoeffs.x / pw + m_cocCoeffs.y;
  			float x     = px/pw + cocr * s.u;
  			float y     = py/pw + cocr * s.v;
  			float dx    = x - s.x;
  			float dy    = y - s.y;
  			float dist2 = dx*dx + dy*dy;
  			if (dist2 <= radius*radius)
			{
				empty = false;
				break;
			}
		}
		if (!empty)
			leaves[lcount++] = leaves[i];
	}
	leaves.resize(lcount);

	// if no leaves remain, consult 2R mode
	if (!leaves.getSize())
		return reconstructCPUFailed(id);

	// process first surface
	Vec4f wbox;
	bool  multi = false;
	Vec4f color(0.f);
	wbox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
	for(int i = 0; i < leaves.getSize(); i++)
	{
  		const CudaNode& node = (*m_nodes)[leaves[i]];

		// test previous leaves of the currently open surface
		for (int j = 0; j < i; j++)
		{
			float RANGE = 1.0f/sqrtf(128.0f);
			Vec3f uvtmin = Vec3f(s.u-RANGE, s.v-RANGE, s.t - 0.5f*RANGE);
			Vec3f uvtmax = Vec3f(s.u+RANGE, s.v+RANGE, s.t + 0.5f*RANGE);
			const CudaNode& prev = (*m_nodes)[leaves[j]];
			if (!sameSurface(node, prev, uvtmin, uvtmax))
			{
				multi = true;
				break; // break as soon as conflict is found
			}
		}
		if (multi)
			break;

		// process points in the node
		for (int j=node.idx0; j < node.idx1; j++)
		{
  			// Reproject to output sample's (u,v,t)
  			const CudaPoint& p = (*m_points)[j];
  
  			float p0x   = p.x * p.w;
  			float p0y   = p.y * p.w;
  			float p0w   = p.w;
  			float px    = p0x + (s.t - p.t) * p.mvx;
  			float py    = p0y + (s.t - p.t) * p.mvy;
  			float pw    = p0w + (s.t - p.t) * p.mvw;
  			float cocr  = m_cocCoeffs.x / pw + m_cocCoeffs.y;
  			float x     = px/pw + cocr * s.u;
  			float y     = py/pw + cocr * s.v;
  			float dx    = x - s.x;
  			float dy    = y - s.y;
  			float dist2 = dx*dx + dy*dy;
  			if (dist2 <= radius*radius)
			{
				float k = dx/dy;
				if (dy >= 0.f)	wbox[0] = min(wbox[0], k),	wbox[1] = max(wbox[1], k);
				else			wbox[2] = min(wbox[2], k),	wbox[3] = max(wbox[3], k);

				// accumulate color using tent filter
				float w = max(0.f, 1.f - sqrtf(dist2) / radius);
				color += w * p.c;
			}
		}
	}

	// we may have certain coverage from the first surface
	if (color.w > 0.f && wbox[1] > wbox[2] && wbox[0] < wbox[3])
		return color * (1.f / color.w);

	// multiple surfaces -> go 2R
	if (multi)
		return reconstructCPUFailed(id);
	
	// get nearest within 2R
	return reconstructCPUNearest(id);
}

//-------------------------------------------------------------------------------------------------

Vec4f CudaReconstruction::reconstructCPUNearest(int id)
{
	CudaSample s;
	constructOutputSample(id, s);
	float dispersion = getDispersion();

	for (float rmul = 1.f; rmul <= 16.f; rmul *= 2.f)
	{
		float radius = rmul * dispersion;
		Array<int> leaves;
		collectNodes(leaves, s, radius);

		// find closest point within the leaves
		int minIdx = -1;
		float minDist = FW_F32_MAX;
		for(int i=0; i < leaves.getSize(); i++)
  		{
  			const CudaNode& node = (*m_nodes)[leaves[i]];
  			for(int j=node.idx0; j < node.idx1; j++)
  			{
  				// Reproject to output sample's (u,v,t)
  				const CudaPoint& p = (*m_points)[j];
  				float p0x   = p.x * p.w;
  				float p0y   = p.y * p.w;
  				float p0w   = p.w;
  				float px    = p0x + (s.t - p.t) * p.mvx;
  				float py    = p0y + (s.t - p.t) * p.mvy;
  				float pw    = p0w + (s.t - p.t) * p.mvw;
  				float cocr  = m_cocCoeffs.x / pw + m_cocCoeffs.y;
  				float x     = px/pw + cocr * s.u;
  				float y     = py/pw + cocr * s.v;
  				float dx    = x - s.x;
  				float dy    = y - s.y;
  				float dist2 = dx*dx + dy*dy;
  				if (dist2 <= minDist)
				{
					minDist = dist2;
					minIdx  = j;
				}
			}
		}
		if (minIdx >= 0)
			return (*m_points)[minIdx].c;
	}

	// no dice
	return Vec4f(0.f);
}

//-------------------------------------------------------------------------------------------------

Vec4f CudaReconstruction::reconstructCPUFailed(int id)
{
	CudaSample s;
	constructOutputSample(id, s);
	float dispersion = getDispersion();
	float radius     = 2.f * dispersion;

	// 2R fetch
	Array<int> leaves;
	collectNodes(leaves, s, radius);

	// no leaves, just find nearest sample and hope for the best
	if (!leaves.getSize())
		return reconstructCPUNearest(id);

	// reconstruct one surface at a time
	Array<Vec2f> xy;		// reprojected points in current surface
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
		color = Vec4f(0.f);
		abox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
		wbox = Vec4f(+FW_F32_MAX, -FW_F32_MAX, +FW_F32_MAX, -FW_F32_MAX);
		do
		{
			// do we end the currently open surface here?
  			const CudaNode& node = (*m_nodes)[leaves[lidx]];
			bool endSurface = false;

			// don't consider ending if there's too few samples
			if (numPoints >= MIN_SAMPLES_PER_SURFACE)
			{
				for (int j = sfirst; j < lidx; ++j)
				{
					float RANGE = 1.0f/sqrtf(128.0f);
					Vec3f uvtmin = Vec3f(s.u-RANGE, s.v-RANGE, s.t - 0.5f*RANGE);
					Vec3f uvtmax = Vec3f(s.u+RANGE, s.v+RANGE, s.t + 0.5f*RANGE);
  					const CudaNode& prev = (*m_nodes)[leaves[j]];
					if (!sameSurface(node, prev, uvtmin, uvtmax))
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
			int idx0 = node.idx0;
			int idx1 = node.idx1;
			for (int i=idx0; i < idx1; i++)
			{
				// Reproject to output sample's (u,v,t)
  				const CudaPoint& p = (*m_points)[i];
  				float p0x   = p.x * p.w;
  				float p0y   = p.y * p.w;
  				float p0w   = p.w;
  				float px    = p0x + (s.t - p.t) * p.mvx;
  				float py    = p0y + (s.t - p.t) * p.mvy;
  				float pw    = p0w + (s.t - p.t) * p.mvw;
  				float cocr  = m_cocCoeffs.x / pw + m_cocCoeffs.y;
  				float x     = px/pw + cocr * s.u;
  				float y     = py/pw + cocr * s.v;
  				float dx    = x - s.x;
  				float dy    = y - s.y;
  				float dist2 = dx*dx + dy*dy;

				// inside radius?
				if (dist2 <= radius*radius)
				{
					xy.add(Vec2f(dx, dy));
					numPoints++;

					float w = 1.f - sqrtf(dist2) / dispersion;

					float k = dx/dy;
					if (w <= 0.f)
					{
						if (dy >= 0.f)	abox[0] = min(abox[0], k), abox[1] = max(abox[1], k);
						else			abox[2] = min(abox[2], k), abox[3] = max(abox[3], k);
					}
					else
					{
						if (dy >= 0.f)	wbox[0] = min(wbox[0], k),	wbox[1] = max(wbox[1], k);
						else			wbox[2] = min(wbox[2], k),	wbox[3] = max(wbox[3], k);

						color += w * p.c;
					}
				}
			} 

			// ok this leaf is now added to the surface, try the next one
			lidx++;
		} while (lidx < leaves.getSize());
		
		// end of surface here

		// test triangles, but only if not the last surface (out of leaves)
		if (lidx < leaves.getSize())
		{
			abox[0] = min(abox[0], wbox[0]);
			abox[1] = max(abox[1], wbox[1]);
			abox[2] = min(abox[2], wbox[2]);
			abox[3] = max(abox[3], wbox[3]);
			if (abox[1] > abox[2] && abox[0] < abox[3])
			{
				if ((wbox[1] > wbox[2] && wbox[0] < wbox[3]) || checkTriangles(xy, dispersion))
					break;
			}
		}

		// end if out of leaves
	} while(lidx < leaves.getSize());

	// all done, either we have a color or we resort to nearest find
	if (color.w > 0.f)
		return color * (1.f / color.w);
	else
		return reconstructCPUNearest(id);
}

//-------------------------------------------------------------------------------------------------

void TreeGather::cudaReconstruction(Image& resultImage, const CameraParams& params)
{
	FW_UNREF(params);
	resultImage.clear(0xff884422);

	Vec2i size = resultImage.getSize();

	// copy output samples
	Array<CudaSample> outputSamples(0, NUM_PATTERNS * m_outputSpp);
	for (int i=0; i < NUM_PATTERNS * m_outputSpp; i++)
	{
		CudaSample& osmp = outputSamples[i];
		Sample&     ismp = m_outputSamples[i];
		osmp.x = ismp.xy.x;
		osmp.y = ismp.xy.y;
		osmp.u = ismp.uv.x;
		osmp.v = ismp.uv.y;
		osmp.t = ismp.t;
	}

	// copy point tree
	Array<CudaNode> nodes(0, m_hierarchy.getSize());
	Array<CudaTraversalNode> tnodes(0, m_hierarchy.getSize());
	for (int i=0; i < m_hierarchy.getSize(); i++)
	{
		Node& inode     = m_hierarchy[i];
		CudaNode& onode = nodes[i];
		CudaTraversalNode& tnode = tnodes[i];

		if (inode.isLeaf())
		{
			onode.idx0 = inode.s0;
			onode.idx1 = inode.s1;
		} else
		{
			onode.idx0 = inode.child0;
			onode.idx1 = inode.child1;
			if (onode.idx0 < onode.idx1)
				swap(onode.idx0, onode.idx1);
		}

		onode.planes[0]	= inode.tlb.planes[0];
		onode.planes[1]	= inode.tlb.planes[1];
		onode.planes[2]	= inode.tlb.planes[2];
		onode.planes[3]	= inode.tlb.planes[3];
		onode.tspan		= inode.tlb.tspan;

		tnode.planes0 = Vec4f(onode.planes[0].x, onode.planes[0].y, onode.planes[0].z, onode.planes[1].x);
		tnode.planes1 = Vec4f(onode.planes[1].y, onode.planes[1].z, onode.planes[2].x, onode.planes[2].y);
		tnode.planes2 = Vec4f(onode.planes[2].z, onode.planes[3].x, onode.planes[3].y, onode.planes[3].z);
		tnode.coc_idx  = Vec4f(0.f, 0.f, bitsToFloat(onode.idx0), bitsToFloat(onode.idx1));
	}

	// copy points
	Array<CudaPoint> points(0, m_samples.getSize());
	for (int i=0; i < m_samples.getSize(); i++)
	{
		Sample&	   ipnt = m_samples[i];
		CudaPoint& opnt = points[i];

		opnt.x = ipnt.xy.x;
		opnt.y = ipnt.xy.y;
		opnt.w = ipnt.w;
		opnt.t = ipnt.t;
		opnt.mvx = ipnt.mv.x;
		opnt.mvy = ipnt.mv.y;
		opnt.mvw = ipnt.mv.z;
		opnt.c = ipnt.color;
	}

	// space for results
	Array<Vec4f> results(0, size.x * size.y);

	// run cuda reconstruction
	CudaReconstruction cr;
	int outputSpp = m_outputSpp;
	cr.init(m_spp, outputSpp, NUM_PATTERNS, size, resultImage, outputSamples, results, nodes, m_rootIndex, tnodes, points, m_cocCoeffs);
//	cr.reconstructCPU();
	cr.reconstructGPU();

	printf("CUDA reconstruction done.\n");
}
