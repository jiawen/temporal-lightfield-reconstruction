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
#include "base/Math.hpp"
#include "base/Random.hpp"

namespace FW
{
Vec2f ToUnitDisk				(const Vec2f& onSquare);
Vec2f FromUnitDisk				(const Vec2f& onDisk);

float hammersley				(int i, int num);		// 0 <= i < n
float halton					(int base, int i);
float sobol						(int dim, int i);
Vec2f sobol2D					(int i);
float larcherPillichshammer		(int i,U32 randomScramble=0);

// Compute the Jacobian between the view and light
// input point is (xw,yw,w) in view's clip space, derivatives are w.r.t. the view pixel coordinates
Mat2f viewLightJacobianAnalytic	(const Mat4f& viewProjection, const Mat4f& viewWorldToCamera, const Mat4f& cameraProjectedZfromW, const Mat4f& lightWorldToClip, Vec2i viewSize, Vec2i lightViewSize, const Vec3f& p, float dwdx, float dwdy);
float lightSpaceDot(const Vec4f& xyw,const Vec2f& dwduv,const Mat4f& cameraProjectedZfromW,const Mat4f& invws,const Mat4f& cameraToShadow2,const Mat4f& cameraToShadow2InvT,const Mat4f& invCameraProjection,const Vec2i& viewSize,Vec3f* viewspaceNormal=NULL);

// p = (x y w) must be in CLIP SPACE (ie. viewport scaling must have been removed)
Vec3f viewSpaceNormalFromWGradients( const Vec3f& p, float dwdu, float dwdv, const Vec2i& viewSize, const Mat4f& invProjection );

template <class T>
void permute(Random& random, Array<T>& data, int lo, int hi)
{
	const int N = hi-lo;
	for(int j=N;j>=2;j--)					// Permutation algo from wikipedia
	{
		int a = random.getS32(j);
		int b = j-1;
		swap(data[lo+a],data[lo+b]);
	}
}

inline bool isPowerOfTwo( U32 v )	{ return (v&(v-1)) == 0; }
inline U32  roundUpToNearestPowerOfTwo( U32 v )
{
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      return( v + 1 );
}

template<class Entry>
struct Sort
{
	static void increasing(Array<Entry>& list, int start=0,int end=0)	{ FW::sort(start,end?end:list.getSize(),list.getPtr(), compareFuncInc, swapFunc); }
	static void decreasing(Array<Entry>& list, int start=0,int end=0)	{ FW::sort(start,end?end:list.getSize(),list.getPtr(), compareFuncDec, swapFunc); }
	static int  compareFuncInc(void* data, int idxA, int idxB)	{ Entry* e=(Entry*)data; return (e[idxA].key < e[idxB].key) ? -1 : (e[idxA].key > e[idxB].key) ? 1 : 0; }
	static int  compareFuncDec(void* data, int idxA, int idxB)	{ Entry* e=(Entry*)data; return (e[idxA].key > e[idxB].key) ? -1 : (e[idxA].key < e[idxB].key) ? 1 : 0; }
	static void swapFunc	  (void* data, int idxA, int idxB)	{ Entry* e=(Entry*)data; swap(e[idxA],e[idxB]); }
};


} //