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

#pragma once
#include "common/SampleBuffer.hpp"
#include "common/CameraParams.hpp"
#include "common/TimeBounds.hpp"
#include "common/Util.hpp"
#include "base/MulticoreLauncher.hpp"
#include "base/Sort.hpp"
#include <cfloat>
#include <cstdio>


namespace FW
{

class TreeGather
{
public:
	TreeGather(const UVTSampleBuffer& sbuf, const CameraParams& params, float apertureAdjust = 1.f, float focalDistanceAdjust = 1.f);
	void	reconstructDofMotion		(Image& image, Image* debugImage=NULL);
	void	reconstructShadows			(UVTSampleBuffer* qbuf, Image* debugImage=NULL);
	void	reconstructDofMotionShadows	(Image& image, const TreeGather& shadowTG);

private:

	struct Stats;
	void	init		(const UVTSampleBuffer& sbuf, const CameraParams& params, float apertureAdjust, float focalDistanceAdjust);
	void	printStats	(const Stats& stats) const;

	//----------------------------------------------------------------------
	// Structures and enums
	//----------------------------------------------------------------------

	enum
	{
		XPOS = (1<<0),						// for relative position mask
		YPOS = (1<<1),
		XNEG = (1<<2),
		YNEG = (1<<3),

		NUM_OUTPUT_SAMPLES			= 128,	// default
		NUM_OUTPUT_SAMPLES_OVERRIDE	= 1,	// use this if UVT override (animations)
		NUM_PATTERNS				= 64,
		MAX_LEAF_SIZE				= 48,
	};

	struct Sample
	{
		Sample() { };

		Vec2f	xy;			// 5D coordinates
		Vec2f	uv;
		float	t;
		Vec4f	color;		// used by reconstruction
		Vec3f	mv;
		float	w;
		Vec2f	wg;			// dw/dx, dw/dy @ t=0
		float	density;
		float	key;		// for sorting

		void	reprojectToUVCenter(const Vec2f& cocCoeffs)
		{
			xy = xy - getCocRadius(cocCoeffs,w)*uv;					// affine position @ (u,v)=0
			uv  = 0;
		}

		bool	reprojectToUVTCenter(const Vec2f& cocCoeffs)
		{
			const Vec2f p = xy - getCocRadius(cocCoeffs,w)*uv;		// affine position @ (u,v)=center
			const Vec3f P = p.toHomogeneous()*w;					// homogeneous position (u,v)=center, t
			float newt = 0.5f;
			Vec3f P0(P - t*mv + newt*mv);							// homogeneous position (u,v,t)=center

			if(P0[2] <= 0.f)	// t=0.5 behind the image plane, try t=0.0
			{
				newt = 0.0f;
				P0 = P - t*mv + newt*mv;							// homogeneous position (u,v)=center, t=0
			}

			if(P0[2] <= 0.f)	// t=0.5 AND t=0.0 behind the image plane, try t=1.0
			{
				newt = 1.0f;
				P0 = P - t*mv + newt*mv;							// homogeneous position (u,v)=center, t=1
			}

			if(P0[2] <= 0.f)	// the input data is broken!
				return false;

			w   = P0[2];
			xy  = P0.toCartesian();
			uv  = 0;
			t   = newt;
			key = w;
			return true;
		}
	};

	struct ReconSample
	{
		Vec2f	xy;			// reprojected to output's (u,v,t)
		int		index;		// in m_samples
		int		rpos;		// relative position
		float	key;		// for sorting

		int		getQuadrant() const		{ return rpos & (XPOS|YPOS); }
	};

	struct Stats
	{
		Stats() { memset(this,0,sizeof(Stats)); }
		void	operator+=(const Stats& src)	{ Vec2d* d=(Vec2d*)this; const Vec2d* s=(const Vec2d*)(&src); for(int i=0;i<sizeof(Stats)/sizeof(Vec2d);i++) d[i] += s[i]; }
		void	newOutputSample() const			{ Vec2d* d=(Vec2d*)this; for(int i=0;i<sizeof(Stats)/sizeof(Vec2d);i++) d[i] += Vec2d(0,1); }
		Vec2d	numLeafNodes;			// #leafnodes / output
		Vec2d	numSamplesInLeafNodes;	// #samples in leafnodes / output
		Vec2d	numSamplesWithin1R;		// #samples within 1R / output
		Vec2d	numSamplesFetched;		// #samples fetched / output
		Vec2d	numSurfaces;			// #surfaces / output
		Vec2d	num2R;					// % outputs needing 2R fetch
		Vec2d	numSurfacesMerged;		// % samples that had surfaces merged
		Vec2d	numAtLeastOne;			// % outputs needing "at least 1" adjustment
	};

	struct TimeLensBounds
	{
		TimeLensBounds() { tspan.x=tspan.y=0.f; }
		TimeLensBounds( const Vec3f& pos, const Vec3f& motion, const Vec2f& bbmin, const Vec2f& bbmax, const Vec2f& cocCoeffs )	{ FW::computeTUXPlanes(pos,motion,bbmin,bbmax,cocCoeffs,planes,tspan); }
		TimeLensBounds( const TimeLensBounds& tb0, const TimeLensBounds& tb1)
		{
			if(tb0.isValid() && tb1.isValid())
				FW::mergeTUXPlanes( tb0.planes, tb1.planes, tb0.tspan, tb1.tspan, planes, tspan );
			else if(tb0.isValid())				*this = tb0;
			else if(tb1.isValid())				*this = tb1;
			else								tspan = 0;
		}

		float	evaluate(int idx,float t,float uv) const	{ return t*planes[idx].x + planes[idx].y*uv + planes[idx].z; }
		bool	isValid() const								{ return tspan.x<tspan.y; }

		Vec3f	planes[4];		// xmin,xmax,ymin,ymax hyperplanes
		Vec2f	tspan;			// 
	};

	struct InitialNode
	{
		InitialNode()			{ child0=-1; numSamples=0; }
		bool	isLeaf() const	{ return child0==-1; }
		int		x0,y0;			// inc
		int		x1,y1;			// exc
		int		child0,child1;
		int		numSamples;		// in subtree
	};

	struct Node
	{
		Node()							{ child0=-1; }
		bool	isLeaf() const			{ return child0==-1; }

		float	getExpectedCost() const
		{
			// Expected cost = P * #samples
			//
			// Static scene: P = xy bounds' area
			// 
			// Dof: constant CoC -> no effect because the xy bounds just move around
			//		Coc range [1,3] with uv[-1]: -1,-3  uv[1]: 1,3
			//
			// Motion: constant motion -> no effect because xy bounds just move around

			const int XMIN=0;
			const int XMAX=1;
			const int YMIN=2;
			const int YMAX=3;
			float dx = (tlb.planes[XMAX][2]-tlb.planes[XMIN][2]) + fabs(tlb.planes[XMAX][1] - tlb.planes[XMIN][1]) + fabs(tlb.planes[XMAX][0] - tlb.planes[XMIN][0]);
			float dy = (tlb.planes[YMAX][2]-tlb.planes[YMIN][2]) + fabs(tlb.planes[YMAX][1] - tlb.planes[YMIN][1]) + fabs(tlb.planes[XMAX][0] - tlb.planes[XMIN][0]);
			return dx * dy * ns;
		}

		bool intersect(const Sample& s,float R) const
		{
			const int XMIN=0;
			const int XMAX=1;
			const int YMIN=2;
			const int YMAX=3;
			// evaluate four hyperplanes to get a conservative XY bounding rect for node at sample's t,u,v, dilate by R
			Vec2f mn = Vec2f( tlb.evaluate(XMIN,s.t,s.uv.x), tlb.evaluate(YMIN,s.t,s.uv.y) ) - R;
			Vec2f mx = Vec2f( tlb.evaluate(XMAX,s.t,s.uv.x), tlb.evaluate(YMAX,s.t,s.uv.y) ) + R;

			// check if sample's xy lies within rect
			return (s.xy.x>=mn.x && s.xy.y>=mn.y && s.xy.x<=mx.x && s.xy.y<=mx.y);
		}

		Node(const Node& n0,const Node& n1)
		{
			// merge bounds
			tlb = TimeLensBounds(n0.tlb, n1.tlb);
			ns    = n0.ns + n1.ns;
			child0= -1;
		};

		TimeLensBounds tlb;

		int		child0,child1;
		int		s0,s1;		// sample indices (inc,exc)
		int		ns;			// total number of sample under this node
	};

	//----------------------------------------------------------------------
	// CUDA reconstruction
	//----------------------------------------------------------------------

	void			cudaReconstruction		(Image& resultImage, const CameraParams& params);

	//----------------------------------------------------------------------
	// "Global" variables
	//----------------------------------------------------------------------

	void			generateOutputSamples	(const CameraParams& params);
	void			reprojectToUVTCenter	(void);
	int 			buildInitialRecursive	(int x0,int x1, int y0,int y1);	// initial tree, used for building the actual tree
	struct BuildTask;
	void			buildRecursive			(int nodeIndex, int maxFrontierSize, Array<Node>& frontier, Array<Node>& hierarchy, BuildTask& bt) const;
	void			emitNodes				(bool isRootNode, int maxFrontierSize, Array<Node>& frontier, Array<Node>& hierarchy) const;

	struct BuildTask
	{
		void init(TreeGather* ptr, int index, int frontierLim, int sampleIndex)
		{
			tg=ptr; nodeIndex=index; maxFrontierSize=frontierLim; 
			if(nodeIndex!=-1)
			{
				const InitialNode& in = tg->m_initialHierarchy[nodeIndex];
				currentSampleIndex = sampleIndex;
				hierarchy.setCapacity( (in.x1-in.x0)*(in.y1-in.y0) );	// not actually a bound, but a good guess
			}
			hierarchy.add();	// space for root (index 0)
		}

		Array<Sample>& getSharedSampleArray(void) const { return tg->m_samples; }

		TreeGather* tg;
		int nodeIndex;				
		int maxFrontierSize;		

		Array<Sample>			candidates;	// temp arrays (avoids repeated calls to malloc)
		Array<int>				surface;
		Array<Vec2f>			boxa, boxb;
		Array<Vec2f>			boxaT1, boxbT1;
		Array<TimeLensBounds>	bounds;

		Array<Node>		frontier;	// return parameter

		int currentSampleIndex;		// sample output array is shared (indexed with currentSampleIndex) to avoid a large memcopy
		Array<Node>	hierarchy;		// private output for avoiding conflicts in parallel emission
	};

	static void buildRecursiveDispatcher	(MulticoreLauncher::Task& task) { BuildTask& bt = *(BuildTask*)task.data; bt.tg->buildRecursive(bt.nodeIndex, bt.maxFrontierSize, bt.frontier, bt.hierarchy, bt); }

	// for sorting samples according to first t, then w
	static int sampleCompareFuncInc( void* data, int idxA, int idxB );

	Array<InitialNode>		m_initialHierarchy;
	Array<Node>				m_hierarchy;
	int						m_rootIndex;

	Array<Sample>			m_samples;
	Array<Array<Sample>>	m_reprojected;
	int						m_reprojWidth;
	int						m_reprojHeight;

	const UVTSampleBuffer*		m_sbuf;
	mutable UVTSampleBuffer*	m_qbuf;				// query points
	mutable UVTSampleBuffer*	m_obuf;				// output sample buffer (DEBUG feature)

	int						m_spp;
	Vec2f					m_cocCoeffs;
	ReconstructionMode		m_reconstructionMode;

	Array<Sample>			m_outputSamples;
	int						m_outputSpp;

	const CameraParams*		m_params;
	bool					m_multicore;

	//----------------------------------------------------------------------
	// Filterer (performs reconstruction)
	//----------------------------------------------------------------------

	class Filterer
	{
	public:
		Filterer() : m_tg(NULL)
		{
			m_leafNodes.     setCapacity(128);
			m_traversalStack.setCapacity(128);
			m_surfaces.      setCapacity(32);
		}

		struct Result
		{
			Result()					{ clear(); }
			void	clear()				{ color=0; density=0; w=0; wg=0; }

			Vec2f	xy;
			Vec4f	color;
			float	density;
			float	w;
			Vec2f	wg;
		};

		void	setTreeGather			(const TreeGather* tg)	{ m_tg = tg; }
		bool	enabled					(void) const			{ return m_tg!=NULL; }

		Result	reconstruct				(const Sample& o,float density,bool reconstructShadow, Stats& stats,Vec4f& debugColor);

	private:
		struct Surface
		{
			Surface()			{ cmpIndices.setCapacity(64); samples.setCapacity(64); clear(); }
			void clear (void)	{ minDist=FW_F32_MAX; minIndex=-1; quadrantMask=0; cmpIndices.clear(); samples.clear(); }
			Array<int>			cmpIndices;
			Array<ReconSample>	samples;
			float				minDist;
			int					minIndex;
			int					quadrantMask;
		};

		struct Leaf
		{
			int	nodeIndex;
			float key;
		};

		int		collectInputSamples2	(Array<Surface>& surfaces, const Sample& s,float R,Stats& stats,bool separateSurfaces=true);		// uses "spectrum heuristic"

		Array<Leaf>				m_leafNodes;			// unique for this task (reduces a memory allocations)
		Array<int>				m_traversalStack;		// unique for this task (reduces a memory allocations)
		Array<Surface>			m_surfaces;				// unique for this task (reduces a memory allocations)

		const TreeGather*		m_tg;
		const Sample&			getSample				(int i) const			{ return m_tg->m_samples[i]; }
		const Node&				getNode					(int i) const			{ return m_tg->m_hierarchy[i]; }
		float					getCocRadius			(float w) const			{ return FW::getCocRadius(m_tg->m_cocCoeffs,w); }
		int						getRootIndex			(void) const			{ return m_tg->m_rootIndex; }
		int						getSPP					(void) const			{ return m_tg->m_spp; }
		ReconstructionMode		getReconstructionMode	(void) const			{ return m_tg->m_reconstructionMode; }
		bool					isMulticore				(void) const			{ return m_tg->m_multicore; }

	public:
		const CameraParams&		getParams				(void) const			{ return *(m_tg->m_params); }
		int						getWidth				(void) const			{ return m_tg->m_sbuf->getWidth();  }
		int						getHeight				(void) const			{ return m_tg->m_sbuf->getHeight(); }
		static float			Gaussian				(float x, float stddev, float mean)	{ float e=2.718281828f; return powf(e,-sqr(x-mean)/(2*sqr(stddev))); }
	};

	//----------------------------------------------------------------------
	// Filter task (multi-core)
	//----------------------------------------------------------------------

	class FilterTask
	{
	public:
		static void dispatcher	(MulticoreLauncher::Task& task) { FilterTask* fttask = (FilterTask*)task.data; fttask->process(task.idx); }

		void	init			(const TreeGather* tg)							{ m_tg=tg; m_filterer.setTreeGather(tg); int w=getWidth(); m_outputColors.reset(w); m_debugColors.reset(w); }
		void	initShadow		(const TreeGather* shadowTG,const Mat4f& c2l,const Mat4f& cameraProjectedZfromW,const Mat4f& invws,const Mat4f& cameraToShadow2,const Mat4f& cameraToShadow2InvT,const Mat4f& invCameraProjection,const Mat4f& cameraToWorldInvT)	{ m_shadowFilterer.setTreeGather(shadowTG); m_c2l=c2l; m_cameraProjectedZfromW=cameraProjectedZfromW; m_invws=invws; m_cameraToShadow2=cameraToShadow2; m_cameraToShadow2InvT=cameraToShadow2InvT; m_invCameraProjection=invCameraProjection; m_cameraToWorldInvT=cameraToWorldInvT; }
		void	process			(int scanline);

	private:
		Vec4f	process			(const Vec2i& pixelIndex);
		Vec4f	process2		(const Vec2i& pixelIndex);

		const TreeGather*		m_tg;
		int						getWidth				(void) const			{ return m_tg->m_sbuf->getWidth(); }
		const Sample&			getOutputSample			(int pidx,int i) const	{ return m_tg->m_outputSamples[pidx*getOutputSPP()+i]; }
		int						getOutputSPP			(void) const			{ return m_tg->m_outputSpp; }
			  UVTSampleBuffer*	getQuerySampleBuffer	(void) const			{ return m_tg->m_qbuf; }
			  UVTSampleBuffer*	getOutputSampleBuffer	(void) const			{ return m_tg->m_obuf; }

		bool					haveShadowFilterer		(void) const			{ return m_shadowFilterer.enabled(); }

		Filterer				m_filterer;
		Filterer				m_shadowFilterer;

		Mat4f					m_c2l;
		Mat4f					m_cameraProjectedZfromW;
		Mat4f					m_invws;
		Mat4f					m_cameraToShadow2;
		Mat4f					m_cameraToShadow2InvT;
		Mat4f					m_invCameraProjection;
		Mat4f					m_cameraToWorldInvT;

	public:
		Array<Vec4f>			m_outputColors;			// unique for this task
		Array<Vec4f>			m_debugColors;			// unique for this task
		Stats					m_stats;				// unique for this task
	};

	//-------------------------------------------------------------------------
	// SameSurface heuristic.
	//-------------------------------------------------------------------------

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

	// return true if there are no mixed signs in the operands
	static inline bool ternaryEqual( int a, int b, int c, int d )
	{
		int minv = min(a,b,c,d);
		int maxv = max(a,b,c,d);
		return maxv-minv < 2;
	}

	// This procedure implements the SameSurface heuristic from Sec. 3.2 between two nodes of the BVH used
	// for fast determination of samples that reproject to the vicinity of the reconstruction location.
	// Specifically, it evaluates the xy coordinates of the screen bounding boxes at the four corners of the uv
	// rectangle centered at the reconstruction location using the hyperplanes, and sees that their relative
	// ordering is the same to within a tolerance. It then does a similar test for the t axis,
	// using the center of the uv cube.
	bool sameSurface(const TreeGather::Node& n0,const TreeGather::Node& n1, const Vec3f& uvtmin, const Vec3f& uvtmax ) const
	{
		const float COC_THRESHOLD = 1*0.1f;

		// evaluate U bounds at t between a and b
		float tmid = 0.5f * (uvtmin.z+uvtmax.z);

		const int XMIN=0;
		const int XMAX=1;
		const int YMIN=2;
		const int YMAX=3;

		float Xmin00 = n0.tlb.evaluate( XMIN, tmid, uvtmin.x );
		float Xmin10 = n0.tlb.evaluate( XMIN, tmid, uvtmax.x );
		float Xmax00 = n0.tlb.evaluate( XMAX, tmid, uvtmin.x );
		float Xmax10 = n0.tlb.evaluate( XMAX, tmid, uvtmax.x );

		float Ymin00 = n0.tlb.evaluate( YMIN, tmid, uvtmin.y );
		float Ymin10 = n0.tlb.evaluate( YMIN, tmid, uvtmax.y );
		float Ymax00 = n0.tlb.evaluate( YMAX, tmid, uvtmin.y );
		float Ymax10 = n0.tlb.evaluate( YMAX, tmid, uvtmax.y );

		float Xmin01 = n1.tlb.evaluate( XMIN, tmid, uvtmin.x );
		float Xmin11 = n1.tlb.evaluate( XMIN, tmid, uvtmax.x );
		float Xmax01 = n1.tlb.evaluate( XMAX, tmid, uvtmin.x );
		float Xmax11 = n1.tlb.evaluate( XMAX, tmid, uvtmax.x );

		float Ymin01 = n1.tlb.evaluate( YMIN, tmid, uvtmin.y );
		float Ymin11 = n1.tlb.evaluate( YMIN, tmid, uvtmax.y );
		float Ymax01 = n1.tlb.evaluate( YMAX, tmid, uvtmin.y );
		float Ymax11 = n1.tlb.evaluate( YMAX, tmid, uvtmax.y );

		bool bCoC =
			   ternaryEqual( ternaryCompare( Xmin00, Xmin01, COC_THRESHOLD ), ternaryCompare( Xmin10, Xmin11, COC_THRESHOLD ) ) &&
			   ternaryEqual( ternaryCompare( Ymin00, Ymin01, COC_THRESHOLD ), ternaryCompare( Ymin10, Ymin11, COC_THRESHOLD ) ) && 
			   ternaryEqual( ternaryCompare( Xmax00, Xmax01, COC_THRESHOLD ), ternaryCompare( Xmax10, Xmax11, COC_THRESHOLD ) ) &&
			   ternaryEqual( ternaryCompare( Ymax00, Ymax01, COC_THRESHOLD ), ternaryCompare( Ymax10, Ymax11, COC_THRESHOLD ) );

		if ( !bCoC )
			return false;
							 
		// evaluate XY (without UV) bounds at ta,tb

		float xmin_ta_node0 = n0.tlb.evaluate(XMIN,uvtmin.z,0.0f);
		float xmin_tb_node0 = n0.tlb.evaluate(XMIN,uvtmax.z,0.0f);
		float ymin_ta_node0 = n0.tlb.evaluate(YMIN,uvtmin.z,0.0f);
		float ymin_tb_node0 = n0.tlb.evaluate(YMIN,uvtmax.z,0.0f);
		float xmax_ta_node0 = n0.tlb.evaluate(XMAX,uvtmin.z,0.0f);
		float xmax_tb_node0 = n0.tlb.evaluate(XMAX,uvtmax.z,0.0f);
		float ymax_ta_node0 = n0.tlb.evaluate(YMAX,uvtmin.z,0.0f);
		float ymax_tb_node0 = n0.tlb.evaluate(YMAX,uvtmax.z,0.0f);

		float xmin_ta_node1 = n1.tlb.evaluate(XMIN,uvtmin.z,0.0f);
		float xmin_tb_node1 = n1.tlb.evaluate(XMIN,uvtmax.z,0.0f);
		float ymin_ta_node1 = n1.tlb.evaluate(YMIN,uvtmin.z,0.0f);
		float ymin_tb_node1 = n1.tlb.evaluate(YMIN,uvtmax.z,0.0f);
		float xmax_ta_node1 = n1.tlb.evaluate(XMAX,uvtmin.z,0.0f);
		float xmax_tb_node1 = n1.tlb.evaluate(XMAX,uvtmax.z,0.0f);
		float ymax_ta_node1 = n1.tlb.evaluate(YMAX,uvtmin.z,0.0f);
		float ymax_tb_node1 = n1.tlb.evaluate(YMAX,uvtmax.z,0.0f);

		bool bTime =
			   ternaryEqual( ternaryCompare( xmin_ta_node0, xmin_ta_node1, COC_THRESHOLD ), ternaryCompare( xmin_tb_node0, xmin_tb_node1, COC_THRESHOLD ) ) &&
			   ternaryEqual( ternaryCompare( xmax_ta_node0, xmax_ta_node1, COC_THRESHOLD ), ternaryCompare( xmax_tb_node0, xmax_tb_node1, COC_THRESHOLD ) ) &&
			   ternaryEqual( ternaryCompare( ymin_ta_node0, ymin_ta_node1, COC_THRESHOLD ), ternaryCompare( ymin_tb_node0, ymin_tb_node1, COC_THRESHOLD ) ) &&
			   ternaryEqual( ternaryCompare( ymax_ta_node0, ymax_ta_node1, COC_THRESHOLD ), ternaryCompare( ymax_tb_node0, ymax_tb_node1, COC_THRESHOLD ) );

		return bTime;
	}
};



} //
