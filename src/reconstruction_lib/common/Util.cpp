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
#include "Util.hpp"
#include <stdio.h>

namespace FW
{

// Low-distortion map between square and circle (Shirley & Chiu JGT97)

Vec2f ToUnitDisk( const Vec2f& onSquare ) 
{
	const float PI = 3.1415926535897932384626433832795f;
	float phi, r, u, v;
	float a = 2*onSquare.x-1;
	float b = 2*onSquare.y-1;

	if (a > -b)
	{
		if (a > b)
		{
			r=a;
			phi = (PI/4 ) * (b/a);
		}
		else
		{
			r = b;
			phi = (PI/4) * (2 - (a/b));
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (PI/4) * (4 + (b/a));
		}
		else
		{
			r = -b;
			if (b != 0)	phi = (PI/4) * (6 - (a/b));
			else		phi = 0;
		}
	}
	u = r * (float)cos( phi );
	v = r * (float)sin( phi );
	return Vec2f( u,v );
}

Vec2f FromUnitDisk(const Vec2f& onDisk)
{
	const float PI = 3.1415926535897932384626433832795f;
	float r   = sqrtf(onDisk.x * onDisk.x + onDisk.y * onDisk.y);
	float phi = atan2(onDisk.y, onDisk.x);

	if (phi < -PI/4)
		phi += 2*PI;

	float a, b, x, y;
	if (phi < PI/4)
	{
		a = r;
		b = phi * a / (PI/4);
	}
	else if (phi < 3*PI/4)
	{
		b = r;
		a = -(phi - PI/2) * b / (PI/4);
	}
	else if (phi < 5*PI/4)
	{
		a = -r;
		b = (phi - PI) * a / (PI/4);
	}
	else
	{
		b = -r;
		a = -(phi - 3*PI/2) * b / (PI/4);
	}

	x = (a + 1) / 2;
	y = (b + 1) / 2;
	return Vec2f(x, y);
}

//------------------------------------------------------------------------

// Compute the Jacobian between the view and light
// input point is (xw,yw,w) in view's clip space, derivatives are w.r.t. the view pixel coordinates
// See Appendix B of the paper.
Mat2f viewLightJacobianAnalytic(const Mat4f& viewProjection, const Mat4f& viewWorldToCamera, const Mat4f& cameraProjectedZfromW, const Mat4f& lightWorldToClip, Vec2i viewSize, Vec2i lightViewSize, const Vec3f& p, float dwdu, float dwdv )
{
	float x = p.x;
	float y = p.y;
	float w = p.z;
	float u = x/w;
	float v = y/w;

	// debugdebug
	//Vec4f pClip = cameraProjectedZfromW * Vec4f( x, y, w, 1.0f );

	// convert pixel derivatives to clip space
	dwdu *= viewSize.x / 2.0f;
	dwdv *= viewSize.y / 2.0f;

	Mat3f dM;
	float dxdu = w + u*dwdu;
	float dydu =     v*dwdu;
	float dxdv =     u*dwdv;
	float dydv = w + v*dwdv;
	dM(0,0) = dxdu; dM(0,1) = dxdv; dM(0,2) = x;
	dM(1,0) = dydu; dM(1,1) = dydv; dM(1,2) = y;
	dM(2,0) = dwdu; dM(2,1) = dwdv; dM(2,2) = w;
	Mat3f invdM = dM.inverted();

	//  (  1    0    0    0  )
	//  (  0    1    0    0  )  = S
	//  (  0    0    1    0  )
	//  ( miw0 miw1 miw2  0  )
	Mat4f S;
	S.setRow(3, Vec4f( invdM.getRow(2), 0.0f ) );

	Mat4f A = lightWorldToClip * (viewProjection * viewWorldToCamera).inverted() * cameraProjectedZfromW * S;

	// derivative of a linear rational expression
	Mat2f result;
	result(0,0) = (A(0,0)*(A(3,0)*u + A(3,1)*v + A(3,2)) - A(3,0)*(A(0,0)*u + A(0,1)*v + A(0,2))) / sqr(A(3,0)*u + A(3,1)*v + A(3,2));
	result(0,1) = (A(0,1)*(A(3,0)*u + A(3,1)*v + A(3,2)) - A(3,1)*(A(0,0)*u + A(0,1)*v + A(0,2))) / sqr(A(3,0)*u + A(3,1)*v + A(3,2));
	result(1,0) = (A(1,0)*(A(3,0)*u + A(3,1)*v + A(3,2)) - A(3,0)*(A(1,0)*u + A(1,1)*v + A(1,2))) / sqr(A(3,0)*u + A(3,1)*v + A(3,2));
	result(1,1) = (A(1,1)*(A(3,0)*u + A(3,1)*v + A(3,2)) - A(3,1)*(A(1,0)*u + A(1,1)*v + A(1,2))) / sqr(A(3,0)*u + A(3,1)*v + A(3,2));

	// debugdebug: return 1/w -- OK!
	//result(0,0) = dot( invdM.getRow(2), Vec3f( u, v, 1.0f ) );

	// convert light clip derivatives to light pixel units
	result(0,0) *= float(lightViewSize.x) / float(viewSize.x);
	result(0,1) *= float(lightViewSize.x) / float(viewSize.x);
	result(1,0) *= float(lightViewSize.y) / float(viewSize.y);
	result(1,1) *= float(lightViewSize.y) / float(viewSize.y);

	return result;
}

//------------------------------------------------------------------------

// p = (x y w) must be in CLIP SPACE (ie. viewport scaling must have been removed)
// returns CAMERA SPACE normal (with flipped OpenGL Z!)
Vec3f viewSpaceNormalFromWGradients( const Vec3f& p, float dwdu, float dwdv, const Vec2i& viewSize, const Mat4f& invProjection )
{
	float x = p.x;
	float y = p.y;
	float w = p.z;
	float u = x/w;
	float v = y/w;

	// convert pixel derivatives to clip space
	dwdu *= viewSize.x / 2.0f;
	dwdv *= viewSize.y / 2.0f;

	Vec4f basis1 = Vec4f( w + u*dwdu,     v*dwdu, dwdu, 0.0f );
	Vec4f basis2 = Vec4f(     u*dwdv, w + v*dwdv, dwdv, 0.0f );

	// undo the FOV scale
	Mat4f ip = invProjection;
	ip.setRow( 2, Vec4f( 0,0,-1,0 ) );
	ip.setRow( 3, Vec4f( 0,0,0,1 ) );

	basis1 = ip * basis1;
	basis2 = ip * basis2;
	
	return cross( basis1.getXYZ(), basis2.getXYZ() ).normalized();
}

//------------------------------------------------------------------------

float lightSpaceDot(
	const Vec4f& xyw,
	const Vec2f& dwduv,
	const Mat4f& cameraProjectedZfromW,
	const Mat4f& invws,
	const Mat4f& cameraToShadow2,
	const Mat4f& cameraToShadow2InvT,
	const Mat4f& invCameraProjection,
	const Vec2i& viewSize,
	Vec3f* viewspaceNormal )
{
	Vec4f xyzw = invws * cameraProjectedZfromW * xyw;	// projected [-1,1]
	Vec3f n = viewSpaceNormalFromWGradients( Vec3f( xyzw.getXY(), xyzw.w ), dwduv.x, dwduv.y, viewSize, invCameraProjection );

	// transform normal to light space

	Vec3f nlight = (cameraToShadow2InvT * Vec4f( n, 0 )).getXYZ();
	Vec3f plight = (cameraToShadow2 * invCameraProjection * xyzw).toCartesian();
	float ldotn = -dot( plight.normalized(), nlight.normalized() );
	if ( viewspaceNormal != NULL )
		*viewspaceNormal = n;
	return ldotn;
}

//------------------------------------------------------------------------


float hammersley(int i, int num)
{
	FW_ASSERT(i>=0 && i<num);
	return (i+0.5f) / num;
}

float halton(int base, int i)
{
	FW_ASSERT(i>=0);

	float h = 0.f;
	float half = rcp(float(base));

	while (i)
	{
		int digit = i % base;
		h = h + digit * half;
		i = (i-digit) / base;
		half = half / base;
	}

	return h;
}

unsigned int SobolGeneratingMatrices[] = {
#include "Sobol5.inl"
};

float sobol(int dim, int i)
{
	FW_ASSERT(i >= 0 && dim >= 0 && dim < 5);
	const unsigned int* const matrix = SobolGeneratingMatrices + dim * 32;
	unsigned int result = 0;
	for (unsigned int c = 0; i; i >>= 1, ++c)
		if (i & 1)
			result ^= matrix[c];
	return result * (1.f / (1ULL << 32));
}

Vec2f sobol2D(int i)
{
	FW_ASSERT(i>=0);
	Vec2f result;
	// remaining components by matrix multiplication 
	unsigned int r1 = 0, r2 = 0; 
	for (unsigned int v1 = 1U << 31, v2 = 3U << 30; i; i >>= 1)
	{
		if (i & 1)
		{
			// vector addition of matrix column by XOR
			r1 ^= v1; 
			r2 ^= v2 << 1;
		}
		// update matrix columns 
		v1 |= v1 >> 1; 
		v2 ^= v2 >> 1;
	}
	// map to unit cube [0,1)^3
	result[0] = r1 * (1.f / (1ULL << 32));
	result[1] = r2 * (1.f / (1ULL << 32));
	return result;
}

float larcherPillichshammer(int i,U32 r)
{
	FW_ASSERT(i>=0);
	for(U32 v=1U<<31; i; i>>=1, v|=v>>1)
		if(i&1)
			r^=v;

	return float( (double)r / (double)0x100000000LL);
}

} //