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

//-----------------------------------------------------------------------------------------
// This source file implements Sections 3-5 of the paper
// Lehtinen, Aila, Chen, Laine, and Durand 2011,
// Temporal Light Field Reconstruction for Rendering Distribution Effects,
// ACM Transactions on Graphics 30(4), article 55,
// with the exception of constructing the 5D acceleration hierarchy
// (found in ReconstructionTreeBuilder.cpp).
// See http://research.nvidia.com/publication/temporal-light-field-reconstruction-rendering-distribution-effects
//     http://groups.csail.mit.edu/graphics/tlfr
//-----------------------------------------------------------------------------------------

#pragma warning(disable:4127)		// conditional expression is constant
#include "Reconstruction.hpp"
#include "common/EdgeFunction.hpp"

namespace FW
{

static const float SHADOW_BIAS = 0.01f;

const bool DEBUG_VERBOSE = false;

TreeGather::Filterer::Result TreeGather::Filterer::reconstruct(const Sample& o,float density,bool reconstructShadow, Stats& stats,Vec4f& debugColor)
{
	int numSamples1R = 0;
	int numSamples2R = 0;
	int numSamplesNR = 0;
	int totalNumSamples = 0;
	Array<Surface>& surfaces = m_surfaces;

	//-----------------------------------------------------------------------------------------
	// Dispersion is the radius of largest empty circle.
	// These number have been measured from sampling pattern.
	//-----------------------------------------------------------------------------------------

	float dispersion;
	switch(getSPP())
	{
	default:	fail("TreeGather::Filterer::reconstruct -- unknown SPP");
	case 256:	dispersion = 0.14f; break;	// measured from sampling pattern -- dispersion is halved when #samples quadruples.
	case 128:	dispersion = 0.18f; break;
	case 64:	dispersion = 0.27f; break;
	case 32:	dispersion = 0.37f; break;
	case 16:	dispersion = 0.50f; break;
	case 8:		dispersion = 0.80f; break;
	case 4:		dispersion = 1.10f; break;
	case 2:		dispersion = 1.34f; break;
	case 1:		dispersion = 2.00f; break;
	}

	// account for non-uniform view sample density in light space when using irregular buffers
	if ( reconstructShadow )
		dispersion *= sqrt(density);

	//-----------------------------------------------------------------------------------------
	// Collect samples that reproject to the vicinity of output XY.
	// - Uses 5D binary tree for search.
	//   Nodes give screen bounding rectangles parameterized by hyperplanes in (u,v,t).
	// - Heuristically determines surfaces using sameSurface(). Sorts surfaces front-to-back.
	//-----------------------------------------------------------------------------------------

	// Step 1: Fetch samples within 1R. If there is exactly one surface, we can use circle/splat coverage. 
	// 0  surfaces --> need more information
	// 1  surfaces --> trivial case, use circle
	// 2+ surfaces --> complex case, use triangle (must refetch with 2R), UNLESS the first surface occupies all 4 quadrants (thus guaranteed to cover output if triangulated).

	numSamples1R = collectInputSamples2(surfaces, o, dispersion, stats);

	numSamples2R = 0;
	numSamplesNR = 0;

	const bool fetch2R = surfaces.getSize()==0 || (surfaces[0].quadrantMask!=0xF && surfaces.getSize()>1);
	if(fetch2R)
	{
		// Step 2: Perform 2R fetch for triangulation.

		numSamples2R = collectInputSamples2(surfaces, o, dispersion*2, stats);

		// Step 3: Emergency mode, must find at least 1 sample...
		// - Don't care about surfaces anymore
		// - Widen the filter

		if(numSamples2R==0)
		{
			stats.numAtLeastOne[0]++;
			numSamplesNR = numSamples2R;
			for(int k=0;k<3 && !numSamplesNR;k++)	// This needs to be limited in order to avoid near-infinite slowdowns in corner cases
			{
				dispersion *= 2.f;

				numSamplesNR = collectInputSamples2(surfaces, o,dispersion*2, stats, false);
			}
		}
	}

	stats.num2R[0]			  += (fetch2R) ? 1 : 0;
	stats.numSamplesWithin1R[0] += numSamples1R;

	totalNumSamples = max(numSamples1R,numSamples2R,numSamplesNR);
	if(DEBUG_VERBOSE)
		printf("Collected %d input samples\n", totalNumSamples);

	//-----------------------------------------------------------------------------------------
	// Combine tiny surfaces
	// - wouldn't be able to triangulate them...
	// - this essentially makes the closer surface semi-transparent, hence somewhat justified.
	// - IMPORTANT: Must not merge small occluded surfaces to the visible ones!
	//-----------------------------------------------------------------------------------------

	const int MIN_SAMPLES_PER_SURFACE = 4;	// Arbitrary (4 means four quadrants)
	int numSurfaces = surfaces.getSize();

	if(fetch2R)
	for(int sidx=0;sidx<surfaces.getSize()-1;sidx++)
	{
		Surface& src = surfaces[sidx];
		if(src.samples.getSize() < MIN_SAMPLES_PER_SURFACE)
		{
			Surface& dst = surfaces[sidx+1];
			dst.samples.add( src.samples );
			src.samples.reset(0);
			if(src.minDist < dst.minDist)
			{
				dst.minDist  = src.minDist;
				dst.minIndex = src.minIndex;
			}
			numSurfaces--;
		}
	}

	stats.numSurfaces[0]       += numSurfaces;
	stats.numSurfacesMerged[0] += (numSurfaces != surfaces.getSize()) ? 1 : 0;

	//-----------------------------------------------------------------------------------------
	// Process input samples, one surface at a time.
	//-----------------------------------------------------------------------------------------

	bool found = false;
	Result result;
	float nearestDist  = FW_F32_MAX;
	Vec2i nearestIndex(-1,1);			// surface, sample

	for(int sidx=0;sidx<surfaces.getSize() && !found;sidx++)
	{
		const Surface& surface = surfaces[sidx];
		const Array<ReconSample>& inputSamples = surface.samples;

		if(inputSamples.getSize()==0)
			continue;

		if(DEBUG_VERBOSE)
		{
			printf("surface %d: (% samples)\n", sidx, surface.samples.getSize());
			for(int k=0;k<surface.samples.getSize();k++)
			{
				const ReconSample& r = inputSamples[k];
				printf("dist=%f, i=%f,%f (%s,%s)\n", (r.xy-o.xy).length(), r.xy.x, r.xy.y, (r.xy[0]-o.xy[0])>=0 ? "+" : "-", (r.xy[1]-o.xy[1])>=0 ? "+" : "-");
			}
		}

		// Set nearest index in case we won't find coverage (fallback mechanism).

		if(surface.minDist < nearestDist)
		{
			nearestDist  = surface.minDist;
			nearestIndex = Vec2i(sidx,surface.minIndex);	// in surfaces
		}

		//-------------------------------------------------------------------------------------
		// Test coverage.
		//-------------------------------------------------------------------------------------

		// Trivial case A: we found 1 surface with 1R radius --> every sample is guaranteed to cover output.
		// Trivial case B: "vertex hit" (DISABLED currently)
		// Trivial case C: last surface

		if(!fetch2R || surface.minDist<0.f || sidx==surfaces.getSize()-1)
			found = true;

		// Find a triangle that covers output sample.

		if(!isMulticore())
			profilePush("Triangulation");

		// EMERGENCY CHECK: avoid very long times spent in triangulation.
		// This should never happen, but keeping to avoid near-infinite loops if something goes wrong.

		if(surface.samples.getSize()>=200)
		{
			printf("%d samples in triangulation\n", surface.samples.getSize());
			found = true;
		}

		for(int m=0  ;m<surface.samples.getSize()-2 && !found; m++)
		for(int k=m+1;k<surface.samples.getSize()-1 && !found; k++)
		for(int l=k+1;l<surface.samples.getSize()   && !found; l++)
		{
			const ReconSample& r0 = inputSamples[m];
			const ReconSample& r1 = inputSamples[k];
			const ReconSample& r2 = inputSamples[l];

			// Early out: Make sure the relative positions of input samples are on different side of output.

			const int rpos = inputSamples[m].rpos | inputSamples[k].rpos | inputSamples[l].rpos;
			if(rpos!=(XPOS|XNEG|YPOS|YNEG))
				continue;

			// Is the triangle valid? I.e. would it fit inside a circle of R=dispersion? 
			// We approximate this by testing that each edge would fit within such a circle (edge length <= 2R).
			// In the worst case this accepts triangles whose center is 1.15R away from vertices (equal triangle).

			if((r0.xy-r1.xy).length() > 2*dispersion || (r0.xy-r2.xy).length() > 2*dispersion || (r1.xy-r2.xy).length() > 2*dispersion)
				continue;

			// Is at least one vertex within R of the reconstruction location?

			if((r0.xy-o.xy).length() > dispersion && (r1.xy-o.xy).length() > dispersion && (r2.xy-o.xy).length() > dispersion)
				continue;

			// Test coverage.

			EdgeFunction ef0(r1.xy.x,r1.xy.y, r2.xy.x,r2.xy.y);		// 1->2
			EdgeFunction ef1(r2.xy.x,r2.xy.y, r0.xy.x,r0.xy.y);		// 2->0
			EdgeFunction ef2(r0.xy.x,r0.xy.y, r1.xy.x,r1.xy.y);		// 0->1
			const float dist0 = ef0.evaluate(o.xy.x,o.xy.y);
			const float dist1 = ef1.evaluate(o.xy.x,o.xy.y);
			const float dist2 = ef2.evaluate(o.xy.x,o.xy.y);
			const bool pos0 = ef0.test(dist0);
			const bool pos1 = ef1.test(dist1);
			const bool pos2 = ef2.test(dist2);
			found = (pos0&&pos1&&pos2) || (!pos0&&!pos1&&!pos2);
		} // triangle

		if(!isMulticore())
			profilePop();

		//-------------------------------------------------------------------------------------
		// Filter color from samples of this surface.
		//-------------------------------------------------------------------------------------

		if(found)
		{
			if(!isMulticore())
				profilePush("Filter");

			result.clear();
			for(int k=0;k<surface.samples.getSize();k++)
			{
				const ReconSample& r = inputSamples[k];
				const Sample& s = getSample( r.index );
				const Vec2f dxy = (o.xy-r.xy);
				const float weight = 1 - dxy.length()/dispersion;		// tent filter
				if(weight<=0)
					continue;

				// if reconstructing shadows: perform per-input-sample thresholding
				Vec4f color;
				if (!reconstructShadow)
					color = s.color;
				else
				{
					bool bLight = (o.w-SHADOW_BIAS <= r.key + dot(dxy,s.wg));
					color = bLight ? s.color : Vec4f( 0, 0, 0, s.color.w );
				}

				result.color   += weight*color;
			}

			const float oow = rcp(result.color.w);
			result.color   *= oow;							// normalize
			nearestIndex = Vec2i(sidx,surface.minIndex);	// from this surface (rarely needed, but anyway)

			if(!isMulticore())
				profilePop();
		}

	} // input samples

	//-----------------------------------------------------------------------------------------
	// Set output color.
	//-----------------------------------------------------------------------------------------

	// Debug visualization: #surfaces, etc.

	debugColor = 0.f;
	debugColor[0] = (float)surfaces.getSize();
	debugColor[1] = (float)numSamples1R;

	// Didn't find ANY surfaces... Return an invalid color.

	if(nearestIndex[0] == -1)
	{
		printf(".");
		result.color = Vec4f(0,0,0,0);
		return result;
	}

	// Copy w, wg and density from the nearest sample (interpolated could be mid-air).

	const ReconSample& nearestAtUVT = surfaces[nearestIndex[0]].samples[nearestIndex[1]];
	const Sample& nearest = getSample( nearestAtUVT.index );
	result.density	= nearest.density;
	result.w		= nearestAtUVT.key;		// w @ output t (!!)
	result.wg		= nearest.wg;
	result.xy		= nearestAtUVT.xy;

	// Either didn't find anything or otherwise should use nearest.

	if(result.color.w == 0)
	{
		if(!reconstructShadow)
		{
			result.color = nearest.color;
		}
		else
		{
			const Vec2f dxy = (o.xy-result.xy);
			bool bLight = (o.w-SHADOW_BIAS <= result.w + dot(dxy,result.wg));
			result.color = bLight ? nearest.color : Vec4f( 0, 0, 0, nearest.color.w );
		}
	}

	// Weighting of output samples.

	if(!reconstructShadow)
	{
		if(getParams().lensFilter == LENS_GAUSSIAN)		result.color *= Gaussian(o.uv[0], 0.5f,0) * Gaussian(o.uv[1], 0.5f,0);
		if(getParams().timeFilter == TIME_GAUSSIAN)		result.color *= Gaussian(o.t, 0.25f,0.5f);
	}

	return result;
}

int TreeGather::Filterer::collectInputSamples2(Array<Surface>& surfaces, const Sample& o,float R,Stats& stats,bool /*separateSurfaces*/)
{
	// Collect leaf nodes that are at least partially within R.

	if(!isMulticore())
		profilePush("Tree gather");

	m_leafNodes.clear();
	m_traversalStack.clear();
	m_traversalStack.add( getRootIndex() );

	while(m_traversalStack.getSize()>0)
	{
		const int nodeIndex = m_traversalStack.removeLast();
		const Node& node = getNode(nodeIndex);

		if(node.intersect(o,R))
		{
			if(node.isLeaf())
			{
				Leaf leaf;
				leaf.nodeIndex = nodeIndex;

				const Sample& s = getSample(node.s0);		// from first sample. (u,v,t) = 0.
				leaf.key = s.w + (o.t-s.t)*s.mv[2];			// w @ output t

				m_leafNodes.add( leaf );
			}
			else
			{
				m_traversalStack.add( node.child0 );
				m_traversalStack.add( node.child1 );
			}
		}
	}

	if(!isMulticore())
		profilePop();

	stats.numLeafNodes[0] += m_leafNodes.getSize();

	// Sort the leaf nodes front-to-back. Each leaf has samples from only one surface.

	if(!isMulticore())
		profilePush("Leaf node sort");

	Sort<Leaf>::increasing(m_leafNodes);

	if(!isMulticore())
		profilePop();

	// Copy samples with R directly to the correct order. Also, creates surfaces here.

	if(!isMulticore())
		profilePush("Per-sample R test");

	surfaces.clear();
	int totalNumSamples = 0;
	int surfaceStart = 0;

	// UVT box in which to check for crossings in sameSurface() (cf. Sec. 3.2)
	const float RANGE = 1.0f/sqrt(128.0f);
	Vec3f uvtmin = Vec3f( o.uv.x-RANGE, o.uv.y-RANGE, o.t-0.5f*RANGE );
	Vec3f uvtmax = Vec3f( o.uv.x+RANGE, o.uv.y+RANGE, o.t+0.5f*RANGE );

	// loop over found leaf nodes
	for(int lidx=0;lidx<m_leafNodes.getSize();lidx++)
	{
		bool firstInThisLeaf = true;
		const Node& node = getNode( m_leafNodes[lidx].nodeIndex );
		stats.numSamplesInLeafNodes[0] += node.ns;

		for(int i=node.s0;i<node.s1;i++)
		{
			// Reproject to output sample's (u,v,t)
			const Sample& s = getSample(i);								// (u,v,t) = center.

			Vec2f xy;
			float w;
			if(s.mv!=Vec3f(0))
			{
				const Vec3f P0 = s.xy.toHomogeneous()*s.w;					// homogeneous position (u,v,t) = center
				const Vec3f P(P0 + (o.t-s.t)*s.mv);							// homogeneous position (u,v)=0, t
				w  = P[2];
				xy = P.toCartesian() + getCocRadius(w)*o.uv;				// affine position @ (u,v,t)
			}
			else	// dof-only
			{
				w  = s.w;
				xy = s.xy + getCocRadius(s.w)*o.uv;
			}

			const Vec2f dxy   = xy-o.xy;
			const float dist2 = dxy.lenSqr();
			if(dist2 <= R*R)									// within radius R?
			{
				ReconSample r;
				r.xy    = xy;								// reproject to output sample's (u,v,t)
				r.rpos  = (dxy.x>=0 ? XPOS : XNEG) | (dxy.y>=0 ? YPOS : YNEG);
				r.key   = w;
				r.index = i;								// in m_samples

				// SameSurface Heuristic: separate samples to surfaces.
				
				if(firstInThisLeaf)
				{
					firstInThisLeaf = false;
					bool sameSurf = true;

					// test all previous leaves
					if(!isMulticore())
						profilePush("Same surface");

					for ( int j = surfaceStart; j < lidx && sameSurf; ++j )
					{
						const Node& testNode = getNode( m_leafNodes[j].nodeIndex );
						sameSurf = m_tg->sameSurface( node, testNode, uvtmin, uvtmax );
					}
					if ( !sameSurf || surfaces.getSize()==0 )
					{
						surfaceStart = lidx;
						Surface& surface = surfaces.add();
						surface.clear();
					}

					if(!isMulticore())
						profilePop();
				}

				Surface& surface = surfaces.getLast();
				surface.samples.add(r);
				surface.quadrantMask |= (1<<r.getQuadrant());

				const float dist = sqrt(dist2);
				if(dist < surface.minDist)
				{
					surface.minDist  = dist;
					surface.minIndex = surface.samples.getSize()-1;
				}

				totalNumSamples++;
			}
		}
	}

	if(!isMulticore())
		profilePop();

	stats.numSamplesFetched[0]  += totalNumSamples;
	return totalNumSamples;
}

} // 
