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

int TreeGather::buildInitialRecursive(int x0,int x1, int y0,int y1)
{
	const int dx = x1-x0;
	const int dy = y1-y0;
	FW_ASSERT(dx>0 && dy>0);

	// Append current node.

	const int nodeIndex = m_initialHierarchy.getSize();

	{
		InitialNode node;
		node.x0 = x0;		// inc
		node.x1 = x1;		// exc
		node.y0 = y0;
		node.y1 = y1;
		m_initialHierarchy.add( node );
	}

	// Create a leaf?
	bool leaf = max(dx,dy)==1;
	if(!leaf && dx*dy<=64)		// arbitrary threshold
	{
		int numSamples = 0;
		for(int y=y0;y<y1;y++)
		for(int x=x0;x<x1;x++)
			numSamples += m_reprojected[y*m_reprojWidth+x].getSize();
		leaf = (numSamples <= MAX_LEAF_SIZE);
	}

	int child0,child1;
	int numSamples = 0;
	if(leaf)	// leaf
	{
		child0 = -1;
		child1 = -1;
		for(int y=y0;y<y1;y++)
		for(int x=x0;x<x1;x++)
			numSamples += m_reprojected[y*m_reprojWidth+x].getSize();
	}
	else
	{
		if(dx>dy)
		{
			int xmid = (x1+x0)/2;
			child0 = buildInitialRecursive(x0,xmid, y0,y1);
			child1 = buildInitialRecursive(xmid,x1, y0,y1);
		}
		else
		{
			int ymid = (y1+y0)/2;
			child0 = buildInitialRecursive(x0,x1, y0,ymid);
			child1 = buildInitialRecursive(x0,x1, ymid,y1);
		}
		FW_ASSERT(child0!=-1 && child1!=-1);
		numSamples = m_initialHierarchy[child0].numSamples + m_initialHierarchy[child1].numSamples;
	}

	m_initialHierarchy[nodeIndex].child0 = child0;	// HMM, doesn't work if assigned directly here...
	m_initialHierarchy[nodeIndex].child1 = child1;
	m_initialHierarchy[nodeIndex].numSamples = numSamples;
	return nodeIndex;
}

// for lexicographic sorting of samples according to t, when w
int TreeGather::sampleCompareFuncInc(void* data, int idxA, int idxB)
{
	const TreeGather::Sample* sa = (const Sample*)data + idxA;
	const TreeGather::Sample* sb = (const Sample*)data + idxB;

	if ( sa->t < sb->t )
		return -1;
	else if ( sa->t > sb->t )
		return 1;
	else
	{
		if ( sa->w < sb->w )
			return -1;
		else if ( sa->w > sb->w )
			return 1;
		else
			return 0;
	}
}

void TreeGather::buildRecursive(int nodeIndex, int maxFrontierSize, Array<Node>& frontier, Array<Node>& hierarchy, BuildTask& bt) const
{
	if(nodeIndex==-1)
		return;

	Array<Sample>& samples = bt.getSharedSampleArray();
	int& currentSampleIndex = bt.currentSampleIndex;

	if(m_initialHierarchy[nodeIndex].isLeaf())
	{
		//profilePush( "Construct leafs" );

		// Collect reprojected samples to a candidate list.

		Array<Sample>& candidates = bt.candidates;
		candidates.clear();

		const InitialNode& in = m_initialHierarchy[nodeIndex];
		for(int y=in.y0;y<in.y1;y++)
		for(int x=in.x0;x<in.x1;x++)
		for(int i=0;i<m_reprojected[y*m_reprojWidth+x].getSize();i++)
			candidates.add( m_reprojected[y*m_reprojWidth+x][i] );

		if ( candidates.getSize() == 0 )
		{
			//profilePop();
			return;
		}

		const Vec2f bbmin(0,0);
		const Vec2f bbmax((float)m_sbuf->getWidth(),(float)m_sbuf->getHeight());

		//profilePush( "sort" );
		// lexicographic sort according to t, then w
		FW::sort( 0, candidates.getSize(), candidates.getPtr(), sampleCompareFuncInc, Sort<Sample>::swapFunc );
		//profilePop();

		//profilePush( "computeCoCs" );
		// compute CoCs and screen bounding boxes for samples
		Array<int>& surface				= bt.surface;
		Array<Vec2f>& boxa				= bt.boxa;
		Array<Vec2f>& boxb				= bt.boxb;
		Array<Vec2f>& boxaT1			= bt.boxaT1;
		Array<Vec2f>& boxbT1			= bt.boxbT1;
		Array<TimeLensBounds>& bounds	= bt.bounds;
		boxa.resize( candidates.getSize() );
		boxb.resize( candidates.getSize() );
		boxaT1.resize( candidates.getSize() );
		boxbT1.resize( candidates.getSize() );
		bounds.resize( candidates.getSize() );
		surface.resize( candidates.getSize() );
		for ( int i = 0; i < candidates.getSize(); ++i )
		{
			const Sample& s = candidates[ i ];

			// construct XYUVT bounding hyperplanes for sample
			Vec3f xywt0 = Vec3f( s.xy*s.w, s.w ) - s.t*s.mv;
			bounds[ i ] = TimeLensBounds( xywt0, s.mv, bbmin, bbmax, m_cocCoeffs );

			// Extrapolate valid t=0 and t=1 screen positions from screen positions computed
			// at the ends of the valid time span. This is a heuristic to avoid the
			// singularities associated with samples that cross the w=0 plane during
			// the shutter interval. Proper treatment would use e.g. the bounds from
			// our paper "Clipless Dual-Space Bounds for Faster Stochastic Rasterization".

			if( bounds[ i ].isValid() )
			{
				float ta = bounds[ i ].tspan[ 0 ];	// start of interval
				float tb = bounds[ i ].tspan[ 1 ];	// end of interval

				// screen pos at t=a, t=b
				Vec3f xywta = Vec3f( s.xy*s.w, s.w ) + (ta-s.t)*s.mv;
				Vec3f xywtb = Vec3f( s.xy*s.w, s.w ) + (tb-s.t)*s.mv;
				Vec2f xyta = xywta.toCartesian();	// project
				Vec2f xytb = xywtb.toCartesian();

				// compute affine motion vector from [ta,tb]
				float oodt = 1.0f / (tb-ta);
				float dxdt = (xytb.x - xyta.x) * oodt;
				float dydt = (xytb.y - xyta.y) * oodt;

				// extrapolate to t=0, t=1
				Vec2f xyt0 = xyta - ta * Vec2f( dxdt, dydt );
				Vec2f xyt1 = xytb + (1.0f-tb) * Vec2f( dxdt, dydt );

				// hack: use CoCs computed at t=a, t=b
				// compute circles of confusion (dx/du, dy/dv) at t=a, t=b
				float cocta = m_cocCoeffs[0]/xywta[2] + m_cocCoeffs[1];
				float coctb = m_cocCoeffs[0]/xywtb[2] + m_cocCoeffs[1];
				boxa[ i ] = xyt0 + Vec2f(cocta);
				boxb[ i ] = xyt0 - Vec2f(cocta);
				boxaT1[ i ] = xyt1 + Vec2f(coctb);
				boxbT1[ i ] = xyt1 - Vec2f(coctb);
			}
		}
		//profilePop(); // computeCoCs

	
		// Append reprojected samples samples to a list.
		// Optionally split into multiple leaf nodes if candidate samples "seem different".
		// Scan bounds for each leaf node.
		// Add node to frontier.

		frontier.clear();

		int surfacebegin = 0;
		int currsurface = 0;
		int n = candidates.getSize();
		// initialize first sample to first surface
		surface[ 0 ] = 0;

		bool bResetBounds = true;
		Vec2f cocboundst0, cocboundst1;
		Vec2f mvmin, mvmax;
		float surfacet = 0.0f;

		//profilePush("scanSurfaces");
		for ( int currcandidate = 1; currcandidate < n; ++currcandidate )
		{
			if ( bResetBounds )
			{
				// reconstruct CoC from boxa/b
				cocboundst0 = Vec2f( 0.5f*(boxa[surfacebegin].x - boxb[surfacebegin].x) );
				cocboundst1 = Vec2f( 0.5f*(boxaT1[surfacebegin].x - boxbT1[surfacebegin].x) );
				// and motion vector from t=0 and t=1 positions
				mvmin = Vec2f( 0.5f*(boxaT1[surfacebegin]+boxbT1[surfacebegin]) - 0.5f*(boxa[surfacebegin]+boxb[surfacebegin]) );
				mvmax = mvmin;
				surfacet = candidates[ surfacebegin ].t;	// this surface only consists of samples bucketed at the same time coordinate
				bResetBounds = false;
			}

			bool bConflict = false;
			bool bEarlyOut = false;

			const float COC_THRESHOLD = 1*2.0f;

			// See that the current sample has been bucketed at the same time instant than the current surface
			// If not, start a new surface.
			if ( surfacet != candidates[ currcandidate ].t )
			{
				surfacebegin = currcandidate;
				surface[ currcandidate ] = ++currsurface;
				bResetBounds = true;
				continue;
			}

			// early out test:
			// if the incoming slope is not too different from the ones already in
			// the surface, the fuzzy ternary compare can't fail because of intersections.
			// even if the slopes are safe, it still might fail because the
			// orientations at t=0 and t=1 are different; hence must check motion magnitude too.
			float coct0 = 0.5f*(boxa[currcandidate].x - boxb[currcandidate].x);
			float coct1 = 0.5f*(boxaT1[currcandidate].x - boxbT1[currcandidate].x);
			Vec2f mv( 0.5f*(boxaT1[currcandidate]+boxbT1[currcandidate]) - 0.5f*(boxa[currcandidate]+boxb[currcandidate]) );
			if ( max( fabs(coct0-cocboundst0[0]), fabs(coct0-cocboundst0[1]) ) < 0.5f*COC_THRESHOLD &&
				 max( fabs(coct1-cocboundst1[0]), fabs(coct1-cocboundst1[1]) ) < 0.5f*COC_THRESHOLD &&
				 max( (mvmin-mv).abs().max(), (mvmax-mv).abs().max() ) < COC_THRESHOLD )
				 bEarlyOut = true;

			// comment this line and uncomment the bIntersect && bEarlyOut test below to test early out
			if ( !bEarlyOut )
			{
				for ( int test = surfacebegin; test < currcandidate; ++test )
				{
					bool bX = ternaryEqual( 
						ternaryCompare( boxa[test].x, boxa[currcandidate].x, COC_THRESHOLD ),	// u==-1, t=0
						ternaryCompare( boxb[test].x, boxb[currcandidate].x, COC_THRESHOLD ),	// u== 1, t=0
						ternaryCompare( boxaT1[test].x, boxaT1[currcandidate].x, COC_THRESHOLD ),	// u==-1, t=1
						ternaryCompare( boxbT1[test].x, boxbT1[currcandidate].x, COC_THRESHOLD ) );	// u==1, t=1

					bool bY = ternaryEqual( 
						ternaryCompare( boxa[test].y, boxa[currcandidate].y, COC_THRESHOLD ),	// v==-1, t=0
						ternaryCompare( boxb[test].y, boxb[currcandidate].y, COC_THRESHOLD ),	// v== 1, t=0
						ternaryCompare( boxaT1[test].y, boxaT1[currcandidate].y, COC_THRESHOLD ),	// v==-1, t=1
						ternaryCompare( boxbT1[test].y, boxbT1[currcandidate].y, COC_THRESHOLD ) );	// v==1, t=1

					bool bIntersect = !(bX && bY);

					// uncomment this to test early out
					/*if ( bIntersect && bEarlyOut )
					{
						Vec2f currmv = Vec2f( 0.5f*(boxaT1[currcandidate]+boxbT1[currcandidate]) - 0.5f*(boxa[currcandidate]+boxb[currcandidate]) );
						Vec2f testmv = Vec2f( 0.5f*(boxaT1[test]+boxbT1[test]) - 0.5f*(boxa[test]+boxb[test]) );

						printf( "no es bueno c0=%.4f, c1=%.4f t0b=[%.4f,%.4f], t1b=[%.4f,%.4f]!\n", coct0, coct1, cocboundst0.x, cocboundst0.y, cocboundst1.x, cocboundst1.y );
						printf( "  incoming: xut0 = [%.4f, %.4f], xut1 = [%.4f, %.4f]\n"\
								"            yvt0 = [%.4f, %.4f], yvt1 = [%.4f, %.4f]\n"\
								"              mv = [%.4f, %.4f]                     \n"\
								"  offender: xut0 = [%.4f, %.4f], xut1 = [%.4f, %.4f]\n"\
								"            yvt0 = [%.4f, %.4f], yvt1 = [%.4f, %.4f]\n"\
								"              mv = [%.4f, %.4f]                     \n",            
								boxa[currcandidate].x, boxb[currcandidate].x, boxaT1[currcandidate].x, boxbT1[currcandidate].x,
								boxa[currcandidate].y, boxb[currcandidate].y, boxaT1[currcandidate].y, boxbT1[currcandidate].y,
								currmv.x, currmv.y,
								boxa[test].x, boxb[test].x, boxaT1[test].x, boxbT1[test].x,
								boxa[test].y, boxb[test].y, boxaT1[test].y, boxbT1[test].y,
								testmv.x, testmv.y );
					}/**/

					if( bIntersect )
					{
						surfacebegin = currcandidate;
						surface[ currcandidate ] = ++currsurface;
						bConflict = true;
						bResetBounds = true;
						break;
					}
				}
			}	// !bEarlyOut
			if ( !bConflict )
			{
				surface[ currcandidate ] = currsurface;

				// enlarge bounds
				cocboundst0[0] = min( coct0, cocboundst0[0] );
				cocboundst0[1] = max( coct0, cocboundst0[0] );
				cocboundst1[0] = min( coct1, cocboundst1[0] );
				cocboundst1[1] = max( coct1, cocboundst1[0] );
				mvmin = FW::min( mvmin, mv );
				mvmax = FW::max( mvmax, mv );
			}
		}
		//profilePop();
		//profilePush("createNodes");

		// phase 2: create nodes
		for ( int i = 0; i < n; /*empty*/ )
		{
			// scan ahead for the next change
			int j;
			for ( j = i+1; j < n && surface[i] == surface[j]; ++j )
				{}

			// cap leaf size
			//j = i + min( j-i, (int)MAX_LEAF_SIZE );

			Node node;
			node.s0				= currentSampleIndex;
			node.child0			= -1;
			node.child1			= -1;

			// add the samples..
			for ( int k = i; k < j; ++k )
			{
				const Sample& s = candidates[ k ];
				samples[currentSampleIndex++] = s;

				// merge sample's XYUVT hyperplanes to node's
				node.tlb = TimeLensBounds(bounds[ k ], node.tlb);
			}

			// ..and finalize the node..
			node.s1 = currentSampleIndex;
			node.ns = node.s1-node.s0;
			FW_ASSERT(node.ns>0);
			frontier.add( node );

			// ..and move on to the next surface.
			i = j;
		}
		//profilePop();	// createNodes


		//profilePop();	// construct leafs
	}
	else
	{
		// Recursive call.

		const InitialNode& in = m_initialHierarchy[nodeIndex];
		Array<Node> frontier1;
		Array<Node> frontier2;
		buildRecursive(in.child0, maxFrontierSize, frontier1, hierarchy, bt);
		buildRecursive(in.child1, maxFrontierSize, frontier2, hierarchy, bt);

		// Merge two frontiers
		frontier.clear();
		frontier.add( frontier1 );
		frontier.add( frontier2 );

		// Create new node(s)
		const int root = 0;
		emitNodes(nodeIndex==root, maxFrontierSize, frontier, hierarchy);
	}
}


// If the frontier is too large, find and emit the best nodes until the frontier is within the threshold.
// Uses an exhaustive O(n^2) algorithm, but n is bounded and small.

void TreeGather::emitNodes(bool isRootNode, int maxFrontierSize, Array<Node>& frontier, Array<Node>& hierarchy) const
{
	const int frontierLim = (isRootNode ? 1 : maxFrontierSize);
	while(frontier.getSize() > frontierLim)
	{
		// Evaluate all pairs.

		float minCost = FW_F32_MAX;
		Vec2i minIndex(-1,-1);
		for(int i=0  ;i<frontier.getSize();i++)
		for(int j=i+1;j<frontier.getSize();j++)
		{
			// cost = area*#samples
			const Node& n0 = frontier[i];
			const Node& n1 = frontier[j];
			Node n01(n0,n1);
			float cost = n01.getExpectedCost() - (n0.getExpectedCost() + n1.getExpectedCost());

			if(cost < minCost)
			{
				minCost  = cost;
				minIndex = Vec2i(i,j);
			}
		}

		// Perform merge. Update frontier.

		const Node& n0 = frontier[minIndex[0]];
		const Node& n1 = frontier[minIndex[1]];
		Node n(n0,n1);
		n.child0 = hierarchy.getSize();	hierarchy.add(n0);
		n.child1 = hierarchy.getSize();	hierarchy.add(n1);

		frontier[minIndex[0]] = n;			// replace n -> n0
		frontier.removeSwap(minIndex[1]);	// remove n1
	}

	if(isRootNode)
	{
		hierarchy[0] = frontier.removeLast();
		FW_ASSERT(frontier.getSize()==0);
	}
}

} // 
