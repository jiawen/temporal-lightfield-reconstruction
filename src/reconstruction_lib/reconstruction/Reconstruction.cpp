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

#pragma warning(disable:4127)		// conditional expression is constant
#include "Reconstruction.hpp"

namespace FW
{

const float LIGHTING_SCALE		= 1.5f;
const float AMBIENT_SCALE		= 0.5f;

static bool g_profile    = false;

//-----------------------------------------------------------------------------
// Ctors.
//-----------------------------------------------------------------------------

TreeGather::TreeGather(const UVTSampleBuffer& sbuf, const CameraParams& params, float apertureAdjust, float focalDistanceAdjust)
{
	init(sbuf,params,apertureAdjust,focalDistanceAdjust);
}

//-----------------------------------------------------------------------------
// Set variables, construct trees etc.
//-----------------------------------------------------------------------------

void TreeGather::init(const UVTSampleBuffer& sbuf, const CameraParams& params, float apertureAdjust, float focalDistanceAdjust)
{
	const int w = sbuf.getWidth();
	const int h = sbuf.getHeight();

	m_sbuf = &sbuf;
	m_obuf = NULL;
	m_qbuf = NULL;
	m_params = &params;
	m_reconstructionMode = params.reconstruction;

	// SPP. irregular --> compute average.

	int spp = sbuf.getNumSamples();
	if(spp==0)
	{
		for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
			spp += sbuf.getNumSamples(x,y);
		spp /=(w*h);
	}
	m_spp = max(1,(int)roundUpToNearestPowerOfTwo(spp));

	// Support for refocus.

    m_cocCoeffs = sbuf.getCocCoeffs();					// used everywhere else
	{
		Vec2f cc = m_cocCoeffs;
		float a = -cc.x;
		float f = -cc.x*rcp(cc.y);
		a *= apertureAdjust;
		f *= focalDistanceAdjust;
		cc.x = -a;
		cc.y = a*rcp(f);
		m_cocCoeffs = cc;
	}
	if(params.overrideRefocusDistance != FW_F32_MAX)
	{
		printf("Experimental refocus enabled\n");
		CameraParams params2;
		memcpy(&params2, &params, sizeof(CameraParams));
		params2.focusDistance  = params2.overrideRefocusDistance;
		m_cocCoeffs = params2.getCocCoeffs();			// compute new coc coefficients
	}

	// Reproject and bucket input samples.

	reprojectToUVTCenter();

	// Build initial hierarchy.

	profilePush("Helper hierarchy");
	const int root = 0;
	m_rootIndex = root;
	m_initialHierarchy.setCapacity(2*m_reprojWidth*m_reprojHeight);	// pre-allocate memory. (n + n/2 + n/4 + n/8 + ... = 2*n)
	buildInitialRecursive(0,m_reprojWidth, 0,m_reprojHeight);

	// Find entry points for parallel builder
	const int numTasksLog2 = 5;
	const int numTasks = 1<<numTasksLog2;
	int entryPoints[numTasks];
	for(int i=0;i<numTasks;i++)
	{
		int node = root;
		for(int j=0;j<numTasksLog2;j++)
		{
			if(node==-1)	// invalid (doesn't exist)
				continue;

			const bool isLeaf = m_initialHierarchy[node].isLeaf();
			if(i&(1<<j)) node = (!isLeaf) ? m_initialHierarchy[node].child1 : -1;
			else		 node = (!isLeaf) ? m_initialHierarchy[node].child0 : node;
		}
		entryPoints[i] = node;
	}
	profilePop();

	// Create parallel tasks (this pins some memory arrays, which would unnecessary in a real system, and hence outside timers)

	profilePush("Pin memory");
	m_samples.reset( m_initialHierarchy[root].numSamples );
	int sampleIndex = 0;

	const int frontierLim = 4;
	BuildTask btask[numTasks];
	for(int i=0;i<numTasks;i++)
	{
		const int ep = entryPoints[i];
		btask[i].init(this, ep, frontierLim, sampleIndex);
		if(ep!=-1)
			sampleIndex += m_initialHierarchy[ btask[i].nodeIndex ].numSamples;
	}
	profilePop();

	// Build actual hierarchy.

	profilePush("Build hierarchy");

	// Launch parallel tasks

	MulticoreLauncher launcher;	// uses #available_cores threads by default
	for(int i=0;i<numTasks;i++)
		launcher.push(buildRecursiveDispatcher, &btask[i]);
	launcher.popAll();

	// Merge sub-hierarchies (we don't know the #nodes per sub-hierarchy in advance)

	int numNodes = 0;
	for(int i=0;i<numTasks;i++)
		numNodes   += btask[i].hierarchy.getSize();

	m_hierarchy.reset(numNodes);

	profilePush("Merge sub-hierarchies");
	int nodeBase = 0;
	for(int i=0;i<numTasks;i++)
	{
		BuildTask& task = btask[i];

		memcpy(m_hierarchy.getPtr(nodeBase), task.hierarchy.getPtr(0), task.hierarchy.getSize()*sizeof(Node));

		for(int j=0;j<task.hierarchy.getSize();j++)
		{
			Node& node = m_hierarchy[nodeBase+j];
			if(node.child0!=-1)	node.child0 += nodeBase;							// private -> global
			if(node.child1!=-1)	node.child1 += nodeBase;							// private -> global
		}

		for(int j=0;j<task.frontier.getSize();j++)
		{
			if(task.frontier[j].child0!=-1)	task.frontier[j].child0 += nodeBase;	// private -> global
			if(task.frontier[j].child1!=-1)	task.frontier[j].child1 += nodeBase;	// private -> global	
		}

		nodeBase   += task.hierarchy.getSize();
	}
	profilePop();

	// Build top of tree

	profilePush("Serial top of tree");
	for(int j=0;j<numTasksLog2;j++)
	{
		int offset = 1<<j;
		for(int i=0;i<numTasks;i+=2*offset)
		{
			btask[i].frontier.add( btask[i+offset].frontier );
			bool isRootNode = (j==numTasksLog2-1);
			emitNodes(isRootNode, frontierLim, btask[i].frontier, m_hierarchy);
		}
	}
	profilePop();

	profilePop();	// actual hierarchy

	// Free memory arrays

	m_reprojected.reset(0);
	m_initialHierarchy.reset(0);

	printf("sizeof(Sample) = %d bytes\n", sizeof(Sample));
	printf("Samples %.1fMB\n", 1.f*m_samples.getSize() * sizeof(Sample) / 1024 / 1024);
	printf("Tree    %.1fMB\n", 1.f*m_hierarchy.getSize() * sizeof(Node) / 1024 / 1024);
}

//-----------------------------------------------------------------------------
// Entry point for defocus and motion
//-----------------------------------------------------------------------------

void TreeGather::reconstructDofMotion(Image& image, Image* debugImage)
{
	if(image.getSize().x < m_sbuf->getWidth() || image.getSize().y < m_sbuf->getHeight())
		fail("TreeGather::reconstructDofMotion image smaller than < sample buffer");

	// Generate output sampling pattern (x,y,u,v,t).

	generateOutputSamples(*m_params);

	// Launch filter tasks. One per scanline.

	// CUDA mode
	if (m_params->enableCuda)
	{
		printf("Filtering on GPU...\n");
		cudaReconstruction(image,*m_params);
		return;
	}

	profilePush("Filtering");

	const int w = m_sbuf->getWidth();
	const int h = m_sbuf->getHeight();

	MulticoreLauncher launcher;
	Array<FilterTask> ftasks;
	ftasks.reset(h);

	m_multicore = !g_profile;

	for(int y=0;y<h;y++)
	{
		FilterTask& ftask = ftasks[y];
		ftask.init(this);

		{
			if(m_multicore)	launcher.push(FilterTask::dispatcher, &ftask, y,1);
			else			ftask.process(y), printf("%d%%\r", 100*y/h);
		}
	}

	if(m_multicore)
		launcher.popAll("Filtering...");
	profilePop();

	// Combine results.

	Stats stats;
	for(int y=0;y<h;y++)
	{
		stats += ftasks[y].m_stats;
		for(int x=0;x<w;x++)
		{
			image.setVec4f(Vec2i(x,y), ftasks[y].m_outputColors[x]);
			if(debugImage)
				debugImage->setVec4f(Vec2i(x,y), ftasks[y].m_debugColors[x]);
		}
	}

	printStats(stats);

	// Output a screenshot.

	image.flipY();
	exportImage("screenshot_DofMotion.png", &image);
	image.flipY();

	// Free memory.

	if(m_obuf)
	{
		m_obuf->serialize("/outputSamples.txt", false);
		delete m_obuf;
	}
}

//-----------------------------------------------------------------------------
// Entry point for shadow-only reconstruction
// -- Not currently callable from the main application.
//-----------------------------------------------------------------------------

void TreeGather::reconstructShadows	(UVTSampleBuffer* qbuf, Image* debugImage)
{
	// This triggers shadow reconstruction internally.

	m_qbuf = qbuf;

	// Launch filter tasks. One per scanline.

	profilePush("Filtering");

	const int w = m_sbuf->getWidth();
	const int h = m_sbuf->getHeight();

	MulticoreLauncher launcher;
	Array<FilterTask> ftasks;
	ftasks.reset(h);

	m_multicore = !g_profile;

	for(int y=0;y<h;y++)
	{
		FilterTask& ftask = ftasks[y];
		ftask.init(this);

		{
			if(m_multicore)	launcher.push(FilterTask::dispatcher, &ftask, y,1);
			else			ftask.process(y), printf("%d%%\r", 100*y/h);
		}
	}

	if(m_multicore)
		launcher.popAll("Filtering...");
	profilePop();

	// Combine results.

	Stats stats;
	for(int y=0;y<h;y++)
	{
		stats += ftasks[y].m_stats;
		for(int x=0;x<w;x++)
			if(debugImage)
				debugImage->setVec4f(Vec2i(x,y), ftasks[y].m_debugColors[x]);
	}

	printStats(stats);
}

//-----------------------------------------------------------------------------
// Entry point for motion+dof with shadows
// -- Not currently callable from the main application.
//-----------------------------------------------------------------------------

void TreeGather::reconstructDofMotionShadows	(Image& image, const TreeGather& shadowTG)
{
	if(image.getSize().x < m_sbuf->getWidth() || image.getSize().y < m_sbuf->getHeight())
		fail("TreeGather::reconstructDofMotionShadows image smaller than < sample buffer");

	// Generate output sampling pattern (x,y,u,v,t) for primary.

	generateOutputSamples(*m_params);

	// Matrices.

	const int w = m_sbuf->getWidth();
	const int h = m_sbuf->getHeight();
	const CameraParams& cameraParams = *m_params;
	const CameraParams& shadowParams = *shadowTG.m_params;

	const Mat4f shadowProjection = shadowParams.projection;
	const Mat4f cameraProjection = cameraParams.projection;
	const Mat4f invCameraProjection = cameraProjection.inverted();
	const Mat4f cameraToShadow  = shadowProjection * shadowParams.camera.getWorldToCamera() * cameraParams.camera.getCameraToWorld() * invCameraProjection;
	const Mat4f cameraToShadow2 = shadowParams.camera.getWorldToCamera() * cameraParams.camera.getCameraToWorld();
	const Mat4f cameraToShadowInvT  = cameraToShadow.inverted().transposed();
	const Mat4f cameraToShadow2InvT = cameraToShadow2.inverted().transposed();
	const Mat4f cameraToWorldInvT = cameraParams.camera.getCameraToWorld().inverted().transposed();
	const Mat4f ws    = Mat4f::scale(Vec3f(0.5f*w, 0.5f*h, 0.5f)) * Mat4f::translate(Vec3f(1.0f));	// [-1,1] -> [window size]
	const Mat4f invws = Mat4f::translate(Vec3f(-1.0f)) * Mat4f::scale(Vec3f(2.f/w, 2.f/h, 2.f));	// [window size] -> [-1,1]

	// cameraProjectedZfromW * (xw yw 1 w) = (xw yw zw w), BUT WITH ZW STILL IN VIEWPORT SCALED UNITS!
	Mat4f cameraProjectedZfromW;
	cameraProjectedZfromW.m22 =  (ws*cameraProjection).m23;
	cameraProjectedZfromW.m23 = -(ws*cameraProjection).m22;

	// This is what the simultaneous reconstruction needs.
	const Mat4f c2l = ws * cameraToShadow * invws * cameraProjectedZfromW;

	// Launch filter tasks. One per scanline.

	profilePush("Filtering");

	MulticoreLauncher launcher;
	Array<FilterTask> ftasks;
	ftasks.reset(h);

	m_multicore = !g_profile;

	for(int y=0;y<h;y++)
	{
		FilterTask& ftask = ftasks[y];
		ftask.init(this);
		ftask.initShadow(&shadowTG,c2l,cameraProjectedZfromW,invws,cameraToShadow2,cameraToShadow2InvT,invCameraProjection,cameraToWorldInvT);

		{
			if(m_multicore)	launcher.push(FilterTask::dispatcher, &ftask, y,1);
			else			ftask.process(y), printf("%d%%\r", 100*y/h);
		}
	}

	if(m_multicore)
		launcher.popAll("Filtering...");
	profilePop();

	// Combine results.

	Stats stats;
	for(int y=0;y<h;y++)
	{
		stats += ftasks[y].m_stats;
		for(int x=0;x<w;x++)
			image.setVec4f(Vec2i(x,y), ftasks[y].m_outputColors[x]);
	}

	printStats(stats);

	// Output a screenshot.

	image.flipY();
	exportImage("screenshot_DofMotionShadows.png", &image);
	image.flipY();
}

//-----------------------------------------------------------------------------
// Internal functions
//-----------------------------------------------------------------------------

void TreeGather::printStats	(const Stats& stats) const
{
	printf("%-6.1f leaf nodes/output sample\n", 1.f*stats.numLeafNodes[0]/stats.numLeafNodes[1]);
	printf("%-6.1f samples in leaf nodes/output sample (avg %.1f/node)\n", 1.f*stats.numSamplesInLeafNodes[0]/stats.numSamplesInLeafNodes[1], 1.f*stats.numSamplesInLeafNodes[0]/stats.numLeafNodes[0]);
	printf("%-6.1f samples within 1R/output sample\n", 1.f*stats.numSamplesWithin1R[0]/stats.numSamplesWithin1R[1]);
	printf("%-5.2f%% output samples did 2R fetch\n", 100.f*stats.num2R[0]/stats.num2R[1]);
	printf("%-6.1f samples fetched/output sample\n", 1.f*stats.numSamplesFetched[0]/stats.numSamplesFetched[1]);
	printf("%-6.2f surfaces/output sample\n", 1.f*stats.numSurfaces[0]/stats.numSurfaces[1]);
	printf("%-5.2f%% output samples did surface merging\n", 100.f*stats.numSurfacesMerged[0]/stats.numSurfacesMerged[1]);
	if(stats.numAtLeastOne[0])
		printf("%-5.2f%% output samples invoked 'at least one'\n", 100.f*stats.numAtLeastOne[0]/stats.numAtLeastOne[1]);
}

void TreeGather::generateOutputSamples(const CameraParams& params)
{
	const bool overrideUVT = (params.overrideUVT != Vec3f(FW_F32_MAX));
	m_outputSpp = (overrideUVT) ? NUM_OUTPUT_SAMPLES_OVERRIDE : NUM_OUTPUT_SAMPLES;
	Random random(242);
	m_outputSamples.reset(0);

	for(int j=0;j<NUM_PATTERNS;j++)
	{
		const Vec2f xyoffset(hammersley(j,NUM_PATTERNS),larcherPillichshammer(j));
		const Vec2f uvoffset(halton(2,j),halton(3,j));
		const float tJitter = random.getF32();

		for(int i=0;i<m_outputSpp;i++)
		{
			const Vec2f s2d = sobol2D(i+1);

			float x = sobol(3,i+1);					// [0,1) inside a pixel, [0,width) for entire image
			float y = larcherPillichshammer(i+1);	// 
			float u = s2d[0];
			float v = s2d[1];
			float t = (i+tJitter)/m_outputSpp;

			Vec2f xy = Vec2f(x,y) + xyoffset;		// Cranley-Patterson rotation (x,y,u,v)
			Vec2f uv = Vec2f(u,v) + uvoffset;
			if(xy[0] >= 1.f) xy[0] -= 1.f;
			if(xy[1] >= 1.f) xy[1] -= 1.f;
			if(uv[0] >= 1.f) uv[0] -= 1.f;
			if(uv[1] >= 1.f) uv[1] -= 1.f;

			// Sorting the samples to increase coherence. Will be useful for CUDA implementation.

			int ub = (int)floor(256*uv[0]);
			int vb = (int)floor(256*uv[1]);
			int tb = (int)floor(256*t);
			int key = 0;
			for(int k=0;k<8;k++)
				key |= ((ub>>k)&1)<<(3*k+2) | ((vb>>k)&1)<<(3*k+1) | ((tb>>k)&1)<<(3*k+0);

			uv = ToUnitDisk(uv);					// [-1,1)

			if(overrideUVT)							// For producing movies (u,v,t)=constant movies
			{
				uv = params.overrideUVT.getXY();
				t  = params.overrideUVT.z;
			}

			if(m_outputSpp==1)						// Special hack
				xy = 0.5f;

			Sample s;
			s.xy = xy;
			s.uv = uv;
			s.t  = t;
			s.key = (float)key;
			m_outputSamples.add(s);
		}

		Sort<Sample>::increasing(m_outputSamples, j*m_outputSpp, (j+1)*m_outputSpp);
	}
}

void TreeGather::reprojectToUVTCenter(void)
{
	profilePush("Init to (u,v,t)=c");
	printf("Init to (u,v,t) center\n");

	const Vec2f cocCoeffs0 = m_sbuf->getCocCoeffs();
	const int w = m_sbuf->getWidth ();
	const int h = m_sbuf->getHeight();
	const int CID_LIGHTSPACE_DENSITY = m_sbuf->getChannelID( "LIGHTSPACE_DENSITY" );	// NOTE: would be separate for each light source

	// Reproject samples to UVT center (the input samples could have been in this format)

	int numInputSamples = 0;	// NOTE: the sample buffer could know this
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
		numInputSamples += m_sbuf->getNumSamples(x,y);

	Array<Sample> samples;
	samples.reset(numInputSamples);

	int sidx=0;
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<m_sbuf->getNumSamples(x,y);i++)
	{
		Sample& s = samples[sidx++];
		s.xy    = m_sbuf->getSampleXY(x,y,i);
		s.uv    = m_sbuf->getSampleUV(x,y,i);
		s.t     = m_sbuf->getSampleT (x,y,i);
		s.color	= m_sbuf->getSampleColor(x,y,i);
		s.mv	= m_sbuf->getSampleMV(x,y,i);
		s.w     = m_sbuf->getSampleW (x,y,i);
		s.wg    = m_sbuf->getSampleWG(x,y,i);
		s.density = (CID_LIGHTSPACE_DENSITY!=-1) ? m_sbuf->getSampleExtra<float>(CID_LIGHTSPACE_DENSITY,x,y,i) : 1.f;

		if(!s.reprojectToUVTCenter(cocCoeffs0))
			sidx--;										// input data is broken, discard sample (w<=0 for entire timespan)
	}

	// Step 1: find bounds (x,y,numHits)

	Vec2f		bbmin(FW_F32_MAX,FW_F32_MAX);
	Vec2f		bbmax(-FW_F32_MAX,-FW_F32_MAX);
	Array<int>	hitCounts;
	hitCounts.reset(w*h);
	memset(hitCounts.getPtr(),0,hitCounts.getNumBytes());

	sidx = 0;
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<m_sbuf->getNumSamples(x,y);i++)
	{
		const Sample& s = samples[sidx++];

		bbmin = min(bbmin, s.xy);
		bbmax = max(bbmax, s.xy);

		int sx = (int)floor(s.xy.x);
		int sy = (int)floor(s.xy.y);
		if(sx>=0 && sy>=0 && sx<w && sy<h)
			hitCounts[sy*w+sx]++;
	}

	// Step 2: focus and size the grid. Basically this is needed only for shadows at this point (all samples can fall into very small area on the "focus" plane).

	int maxSamplesPerPixel=0;
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
		maxSamplesPerPixel = max(maxSamplesPerPixel,hitCounts[y*w+x]);
	printf("  bbmin:   %.2f,%.2f\n", bbmin.x,bbmin.y);
	printf("  bbmax:   %.2f,%.2f\n", bbmax.x,bbmax.y);
	printf("  spp max: %d\n", maxSamplesPerPixel);

	if(maxSamplesPerPixel==0)
		fail("All samples were discarded in reprojection to (u,v,t)=0. Invalid params?\n");

	float scale = sqrt(1.f*maxSamplesPerPixel/MAX_LEAF_SIZE);			// How much x and y should be scaled so that maximum leaf size is approx. MAX_LEAF_SIZE
	const int MAX_XY_SIZE = 4096;
	if(scale*(bbmax-bbmin).max() > MAX_XY_SIZE)
		scale = MAX_XY_SIZE / (bbmax-bbmin).max();						// Avoid excessively large surfaces
	bbmin *= scale;
	bbmax *= scale;

	const Vec2i bbminInt((int)floor(bbmin.x), (int)floor(bbmin.y));		// inc
	const Vec2i bbmaxInt((int)floor(bbmax.x)+1, (int)floor(bbmax.y)+1);	// exc

	m_reprojWidth  = bbmaxInt.x - bbminInt.x;
	m_reprojHeight = bbmaxInt.y - bbminInt.y;
	m_reprojected.reset(m_reprojWidth*m_reprojHeight);
	printf("  Reprojection buffer size: %dx%d (scale=%.2f)\n", m_reprojWidth,m_reprojHeight,scale);

	// Step 3: bucket the input samples

	int numSamplesDiscarded = 0;
	int numSamplesAccepted  = 0;

	sidx = 0;
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	for(int i=0;i<m_sbuf->getNumSamples(x,y);i++)
	{
		const Sample& s = samples[sidx++];

		int sx = (int)floor(scale*s.xy.x) - bbminInt.x;
		int sy = (int)floor(scale*s.xy.y) - bbminInt.y;
		if(sx<0 || sy<0 || sx>=m_reprojWidth || sy>=m_reprojHeight)
		{
			numSamplesDiscarded++;
			continue;
		}
		m_reprojected[sy*m_reprojWidth+sx].add( s );
		numSamplesAccepted++;
	}

	maxSamplesPerPixel=0;
	for(int y=0;y<m_reprojHeight;y++)
	for(int x=0;x<m_reprojWidth ;x++)
		maxSamplesPerPixel = max(maxSamplesPerPixel,m_reprojected[y*m_reprojWidth+x].getSize());
	printf("  Max samples/pixel (UVT0): %d\n", maxSamplesPerPixel);

	if(numSamplesDiscarded>0)
		printf("  %d samples discarded in reprojection to (u,v,t)=0\n", numSamplesDiscarded);
	if(numSamplesAccepted==0)
		fail("All samples were discarded in reprojection to (u,v,t)=0. Invalid params?\n");

	profilePop();
}

void TreeGather::FilterTask::process(int y)
{
	for(int x=0;x<getWidth();x++)
		m_outputColors[x] = (haveShadowFilterer()) ? process2(Vec2i(x,y)) : process( Vec2i(x,y) );
}

// does motion+dof, shadow-only
Vec4f TreeGather::FilterTask::process(const Vec2i& pixelIndex)
{
	const int x = pixelIndex.x;
	const int y = pixelIndex.y;
	const int outputPatternIndex = hashBits(x,y) % NUM_PATTERNS;

	UVTSampleBuffer* qbuf = getQuerySampleBuffer();
	UVTSampleBuffer* obuf = getOutputSampleBuffer();

	const bool reconstructShadow = (qbuf!=NULL);
	const int M = reconstructShadow ? qbuf->getNumSamples(x,y) : getOutputSPP();

	int CID_LIGHTSPACE_DENSITY = 0;
	if ( reconstructShadow )
		CID_LIGHTSPACE_DENSITY = qbuf->getChannelID( "LIGHTSPACE_DENSITY" );

	Vec4f pixelColor(0);

	for(int i=0;i<M;i++)
	{
		m_stats.newOutputSample();

		//-----------------------------------------------------------------------------------------
		// Construct ith output sample.
		//-----------------------------------------------------------------------------------------

		Sample o;
		float density = 1.0f;

		if(reconstructShadow)
		{
			o.xy    = qbuf->getSampleXY   			(x,y,i);
			o.uv    = qbuf->getSampleUV   			(x,y,i);
			o.t     = qbuf->getSampleT    			(x,y,i);
			o.color = qbuf->getSampleColor			(x,y,i);
			o.w		= qbuf->getSampleW				(x,y,i);
			o.mv    = qbuf->getSampleMV				(x,y,i);
			o.wg	= qbuf->getSampleWG				(x,y,i);
			density = qbuf->getSampleExtra<float>	(CID_LIGHTSPACE_DENSITY,x,y,i);
		}
		else
		{
			// Position (x,y,u,v,t) into current pixel.
			o = getOutputSample(outputPatternIndex,i);
			o.xy += Vec2f(pixelIndex);
		}

		if(obuf)
		{
			obuf->m_xy[ obuf->getIndex(x,y,i) ] = o.xy;
			obuf->m_uv[ obuf->getIndex(x,y,i) ] = o.uv;
			obuf->m_t [ obuf->getIndex(x,y,i) ] = o.t;
		}

		const Filterer::Result result = m_filterer.reconstruct(o,density,reconstructShadow, m_stats,m_debugColors[x]);

		Vec4f color = result.color;
		pixelColor += color;

		if(reconstructShadow)
			qbuf->setSampleColor(x,y,i, color);

		if(obuf)
			obuf->setSampleColor(x,y,i, color);

	} // output sample

	if(pixelColor.w==0)
		pixelColor = Vec4f(1,0,0,1);

	return pixelColor * rcp(pixelColor.w);
}

// Does primary + shadow
Vec4f TreeGather::FilterTask::process2(const Vec2i& pixelIndex)
{
	const int x = pixelIndex.x;
	const int y = pixelIndex.y;
	const int M = getOutputSPP();
	const int samplingPatternIndex = hashBits(x,y) % NUM_PATTERNS;

	Random random(hashBits(x,y));
	const Vec2f uvoffset(random.getF32(),random.getF32());
	const float camFarW = m_filterer.getParams().camera.getFar() - 1e-2f;
	const Vec2f cCoCCoeffs = m_filterer.getParams().getCocCoeffs();
	const Vec2f lCoCCoeffs = m_shadowFilterer.getParams().getCocCoeffs();
	const int cw = m_filterer.getWidth ();
	const int ch = m_filterer.getHeight();
	const int lw = m_shadowFilterer.getWidth ();
	const int lh = m_shadowFilterer.getHeight();

	if(cw!=lw || ch!=lh)
		fail("TreeGather::FilterTask::process2 -- cw!=lw || ch!=lh");

	Vec4f pixelColor(0);

	for(int i=0;i<M;i++)
	{
		m_stats.newOutputSample();

		// Position (x,y,u,v,t) into current pixel.

		Sample cs = getOutputSample(samplingPatternIndex,i);
		cs.xy += Vec2f(pixelIndex);

		// Recontruct motion + dof from camera. 

		const Filterer::Result result = m_filterer.reconstruct(cs,1.f,false, m_stats,m_debugColors[x]);

		cs.xy      = result.xy;
		cs.color   = result.color;
		cs.density = result.density;
		cs.w       = result.w;
		cs.wg      = result.wg;

		if(cs.color == Vec4f(0,0,0,0))
		{
			pixelColor = Vec4f(1,0,0,1);
			break;
		}

		// Generate shadow "ray".

		Vec4f color = cs.color;

		if(cs.w<=camFarW)
		{
			cs.reprojectToUVCenter(cCoCCoeffs);				// as seen from pinhole camera
			Vec4f lpos = m_c2l * Vec4f(cs.xy*cs.w,1,cs.w);	// pinhole camera -> pinhole light
			lpos = Vec4f(lpos.toCartesian(),lpos.w);
			if(lpos.x>=0 && lpos.y>=0 && lpos.x<=lw && lpos.y<=lh && lpos.w>0)
			{
				Sample ls;
				ls.xy = lpos.getXY();
				ls.w  = lpos.w;
				ls.t  = cs.t;									// to have consistent geometry [0,1]

				Vec2f uv = Vec2f(sobol(3,i+1),sobol(4,i+1)) + uvoffset;		// position on the light source
				if(uv[0] >= 1.f) uv[0] -= 1.f;
				if(uv[1] >= 1.f) uv[1] -= 1.f;
				ls.uv = 2.f*uv-1.f;								// square light source [-1,1]

				ls.xy += ls.uv*getCocRadius(lCoCCoeffs,ls.w);	// NOTE: two-plane parameterization.

				const Filterer::Result shadowResult = m_shadowFilterer.reconstruct(ls,cs.density,true, m_stats,m_debugColors[x]);
				color = shadowResult.color;

				// compute cosine term for light based on W gradients
				Vec3f viewspaceNormal;
				float ldotn = lightSpaceDot(
					Vec4f(cs.xy*cs.w,1,cs.w),		// zy1w
					cs.wg,							// w-gradient
					m_cameraProjectedZfromW,
					m_invws,
					m_cameraToShadow2,
					m_cameraToShadow2InvT,
					m_invCameraProjection,
					Vec2i( cw,ch ),
					&viewspaceNormal );

				// compute ambient from world space normal
				Vec3f worldspaceNormal = (m_cameraToWorldInvT * Vec4f( viewspaceNormal, 0 )).getXYZ();
				float ambient = worldspaceNormal.y * 0.5f + 0.5f;

				float cosine = max( ldotn, 0.0f );
				color *= LIGHTING_SCALE * Vec4f( cosine, cosine, cosine, 1 );
				color += ambient*AMBIENT_SCALE*cs.color;
				color.w = 1;
			}
		}

		pixelColor += color;

	} // output sample

	if(pixelColor.w==0)
		pixelColor = Vec4f(1,0,0,1);

	pixelColor *= rcp(pixelColor.w);			// normalize
	return pixelColor;
}

} // 
