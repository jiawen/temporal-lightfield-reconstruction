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

#include "TimeBounds.hpp"
#include <float.h>

namespace FW
{

// internal worker, used separately for x,t and y,t by computePerspectivePlanes
bool computePerspectivePlanes1D( F32 xw, F32 w, F32 dxw, F32 dw, F32 xmin, F32 xmax, Vec2f* pDest, Vec2f* pDestSpan );

// computes planes (well, lines) that bound the perspective motion of one point undergoing 3D affine motion
// pos and motion are homogeneous t0 position and motion vector (xw,yw,w), (dxw,dyw,dw)
// bbmin, bbmax are the x and y bounds of the screen rectangle (i.e., [0 0], [1280 720])
// pDestPlanes must have room for 4 2d vectors, stored in order xmin,xmax,ymin,ymax
// pDestTimeSpan must have room for a single Vec2f that upon termination specifies the valid time range
// returns true if the point is visible at any instant, false otherwise (in which case nothing is written to destination)
// format of output plane p is x = p.x*t + p.y
bool computePerspectivePlanes( const Vec3f& pos, const Vec3f& motion, const Vec2f& bbmin, const Vec2f& bbmax, Vec2f* pDestPlanes, Vec2f& rDestTimeSpan )
{
	Vec2f spanX, spanY;

	bool bx = computePerspectivePlanes1D( pos.x, pos.z, motion.x, motion.z, bbmin.x, bbmax.x, pDestPlanes, &spanX );
	bool by = computePerspectivePlanes1D( pos.y, pos.z, motion.y, motion.z, bbmin.y, bbmax.y, pDestPlanes+2, &spanY );

	// if either x or y is completely outside,
	// not visible at all
	if ( !bx || !by )
		return false;

	// intersect time spans from X&Y
	rDestTimeSpan.x = max( spanX.x, spanY.x );
	rDestTimeSpan.y = min( spanX.y, spanY.y );

	// if the ranges don't overlap, the bounds invert ==> point not visible
	return rDestTimeSpan.x < rDestTimeSpan.y;
}

// computes XYUVT hyperplanes that bound the perspective motion of one point undergoing 3D affine motion
// pos and motion are homogeneous t0 position and motion vector (xw,yw,w), (dxw,dyw,dw)
// bbmin, bbmax are the x and y bounds of the screen rectangle (i.e., [0 0], [1280 720])
// pDestPlanes must have room for 4 3d vectors, stored in order xmin,xmax,ymin,ymax
// pDestTimeSpan must have room for a single Vec2f that upon termination specifies the valid time range
// returns true if the point is visible at any instant, false otherwise (in which case nothing is written to destination)
// format of output plane p is
// x = p.x*t + p.y*u + p.z for the xmin and xmax planes, and
// y = p.x*t + p.y*v + p.z for the ymin and ymax planes.
bool computeTUXPlanes( const Vec3f& pos, const Vec3f& motion, const Vec2f& bbmin, const Vec2f& bbmax, const Vec2f& cocCoeffs, Vec3f* pDestPlanes, Vec2f& rDestTimeSpan )
{
	rDestTimeSpan = Vec2f( 0.0f, 0.0f );

	Vec2f planes[4];
	// first compute valid time spans during which the sample is on screen
	// and bounding lines in XT and YT
	if ( !computePerspectivePlanes( pos, motion, bbmin, bbmax, planes, rDestTimeSpan ) )
		return false;

	// then lift up to full XTU and YTV planes

	float fTimes[3];
	fTimes[0] = rDestTimeSpan[0];
	fTimes[1] = rDestTimeSpan[1];
	fTimes[2] = 0.5f*(rDestTimeSpan[0]+rDestTimeSpan[1]);

	// evaluate (x,y) at ends of t span (and middle)
	Vec3f xyw[3];
	xyw[0] = (pos + fTimes[0]*motion);							// t=start
	xyw[1] = (pos + fTimes[1]*motion);							// t=end
	xyw[2] = (pos + fTimes[2]*motion);	// t=middle

	Vec2f endpoints[3];	// projected screen coordinates at (tstart,tend,tmiddle)
	endpoints[0] = xyw[0].toCartesian();	// u=v=0, t=start
	endpoints[1] = xyw[1].toCartesian();	// u=v=0, t=end
	endpoints[2] = xyw[2].toCartesian();	// u=v=0, t=middle

	float fCocs[3];
	fCocs[0] = cocCoeffs[0]/xyw[0].z + cocCoeffs[1];
	fCocs[1] = cocCoeffs[0]/xyw[1].z + cocCoeffs[1];
	fCocs[2] = cocCoeffs[0]/xyw[2].z + cocCoeffs[1];

	float fAvgCoC = (1.f/3.f)*(fCocs[0]+fCocs[1]+fCocs[2]);

	// t slope stays unchanged, turns out least squares u slope is just average of the individual slopes dx/du (=CoC) at tstart, tmiddle, tend
	pDestPlanes[0] = Vec3f( planes[0].x, fAvgCoC, FLT_MAX );
	pDestPlanes[1] = Vec3f( planes[1].x, fAvgCoC, -FLT_MAX );
	pDestPlanes[2] = Vec3f( planes[2].x, fAvgCoC, FLT_MAX );
	pDestPlanes[3] = Vec3f( planes[3].x, fAvgCoC, -FLT_MAX );

	// evaluate the constant term C for the pleq so that it bounds the samples from below
	for ( int i = 0; i < 3; ++i )
	{
		// the pleq's form is x = A*t + B*u (+ C)
		// so when evaluating the constant, we compute 1*x-A*t-B*u

		// x, t, u=-1
		float fdotx11 = 1.0f * (endpoints[i].x - fCocs[i]) - planes[0].x * fTimes[i] - fAvgCoC * (-1.0f);
		// x, t, u=1
		float fdotx21 = 1.0f * (endpoints[i].x + fCocs[i]) - planes[0].x * fTimes[i] - fAvgCoC * (1.0f);

		// x, t, u=-1
		float fdotx12 = 1.0f * (endpoints[i].x - fCocs[i]) - planes[1].x * fTimes[i] - fAvgCoC * (-1.0f);
		// x, t, u=1
		float fdotx22 = 1.0f * (endpoints[i].x + fCocs[i]) - planes[1].x * fTimes[i] - fAvgCoC * (1.0f);

		// x, t, u=-1
		float fdoty11 = 1.0f * (endpoints[i].y - fCocs[i]) - planes[2].x * fTimes[i] - fAvgCoC * (-1.0f);
		// x, t, u=1
		float fdoty21 = 1.0f * (endpoints[i].y + fCocs[i]) - planes[2].x * fTimes[i] - fAvgCoC * (1.0f);

		// x, t, u=-1
		float fdoty12 = 1.0f * (endpoints[i].y - fCocs[i]) - planes[3].x * fTimes[i] - fAvgCoC * (-1.0f);
		// x, t, u=1
		float fdoty22 = 1.0f * (endpoints[i].y + fCocs[i]) - planes[3].x * fTimes[i] - fAvgCoC * (1.0f);

		pDestPlanes[0].z = min( pDestPlanes[0].z, fdotx11, fdotx21 );
		pDestPlanes[1].z = max( pDestPlanes[1].z, fdotx12, fdotx22 );
		pDestPlanes[2].z = min( pDestPlanes[2].z, fdoty11, fdoty21 );
		pDestPlanes[3].z = max( pDestPlanes[3].z, fdoty12, fdoty22 );
	}

	return true;
}

// merges two TUX bounding planes, such that the result bounds both from above
// use same code for minimum: pass in -tux1, -tux2, and negate the result
// input planes are of the form x = Bt + Cu + X0
// tux(1) = B, tux(2) = C, tux(3) = X0
// i.e. B == x velocity, C == circle of confusion
Vec3f mergeTUXPlanes( const Vec3f& plane1, const Vec3f& plane2 )
{
	// corners of TU rectangle (0,-1), (1,-1), (1, 1), (0,1)
	// and a last row of ones
	// pts = [0 -1 1; 1 -1 1; 1 1 1; 0 1 1].';

	// --- evaluate X/Ys at the TU/TV corners for both input planes ---
	// x1(1) = B1*0 + C1*-1 + X0, x1(2) = B1*1 + C1*-1 + X0, etc.
	Vec4f x1, x2, maxx;

	x1[0] =           -plane1[1]+plane1[2]; x1[1] = plane1[0]-plane1[1]+plane1[2];	// x at t=0,u=-1 and t=1,u=-1
	x1[2] =  plane1[0]+plane1[1]+plane1[2]; x1[3] =           plane1[1]+plane1[2];	// x at t=1,u=1 and t=0,u=1

	x2[0] =           -plane2[1]+plane2[2]; x2[1] = plane2[0]-plane2[1]+plane2[2];	// x at t=0,u=-1 and t=1,u=-1
	x2[2] =  plane2[0]+plane2[1]+plane2[2]; x2[3] =           plane2[1]+plane2[2];	// x at t=1,u=1 and t=0,u=1

	// --- take maximum X at each TU corner
	maxx[0] = max( x1[0], x2[0] );
	maxx[1] = max( x1[1], x2[1] );
	maxx[2] = max( x1[2], x2[2] );
	maxx[3] = max( x1[3], x2[3] );

	// --- fit new plane to maximums
	// linear regression for normal estimate
    
    // a*t(i) + b*u(i) + c = xi,   i=1,..,4 (each TU corner)
    // Compute quadratic energy functional, take Jacobian w.r.t. [a b c]:
    //                4*a + 4*c - 2*x2 - 2*x3
    //        8*b + 2*x1 + 2*x2 - 2*x3 - 2*x4
    //  4*a + 8*c - 2*x1 - 2*x2 - 2*x3 - 2*x4
    
    // ==>
    //      C1 = 2*(x2+x3)
    //      C2 = 2*(x3+x4-x1-x2)
    //      C3 = 2*(x1+x2+x3+x4)
    //
    //      a = 0.5*C1 - 0.25*C3
    //      b = 0.125*C2
    //      c = -0.25*C1 + 0.25*C3
    
    Vec3f c( maxx[1]+maxx[2], maxx[2]+maxx[3]-maxx[0]-maxx[1], maxx[0]+maxx[1]+maxx[2]+maxx[3] );
    
    float a = c[0] - 0.5f*c[2];
	float b = 0.25f*c[1];
    //c = -0.25*c(1) + 0.25*c(3);    % not needed!
    
    // constants = evaluate a*t + b*u at all corners of the TU rectangle
	Vec4f constants;
	constants[0] =  -b;	// t=0,u=-1
	constants[1] = a-b;	// t=1,u=-1
	constants[2] = a+b;	// t=1,u= 1
	constants[3] =   b;	// t=0,u= 1

    // take maximum
	float d = (maxx-constants).max();

	// done!
	return Vec3f( a, b, d );
}

// merges two TUX/TVY plane sets, X and Y separately
// planes1 and planes2 contain 4 planes each (xmin,xmax,ymin,ymax)
void mergeTUXPlanes( const Vec3f* planes1, const Vec3f* planes2, const Vec2f& span1, const Vec2f& span2, Vec3f* pDestPlanes, Vec2f& rDestTimeSpan )
{
	pDestPlanes[0] = -mergeTUXPlanes( -planes1[0], -planes2[0] );	// negate because xmin bounds from below
	pDestPlanes[1] = mergeTUXPlanes( planes1[1], planes2[1] );
	pDestPlanes[2] = -mergeTUXPlanes( -planes1[2], -planes2[2] );	// negate because ymin bounds from below
	pDestPlanes[3] = mergeTUXPlanes( planes1[3], planes2[3] );
	rDestTimeSpan.x = min( span1.x, span2.x );
	rDestTimeSpan.y = max( span1.y, span2.y );
}


// this function implements the construction from Appendix A.
bool computePerspectivePlanes1D( F32 x0, F32 w0, F32 dx, F32 dw, F32 xmin, F32 xmax, Vec2f* pDest, Vec2f* pDestSpan )
{
	// no 3D motion? just return the line x=x0/w0
	if ( dx==0.0f && dw==0.0f )
	{
		F32 xow0 = x0/w0;
		pDest[0] = Vec2f( 0.0f, xow0 );
		pDest[1] = pDest[0];
		(*pDestSpan)[0] = 0.f;
		(*pDestSpan)[1] = 1.f;

		// see if the point is on the screen
		return w0 > 0.0f; // && xow0 >= xmin && xow0 <= xmax;
	}

	// TODO special case: apparent screen position stays constant (motion
	// from/towards camera)

	// limit scope to the time the point lies within w>0
	// compute time of crossing of the w=0 plane,
	// shrink time bounds as appropriate
	Vec2f tbounds( 0.0f, 1.0f );
	if ( dw != 0.0f )
	{
	    //  w0+t*dw=0   <=>  t=-w0/dw;
	    F32 tw0 = -w0/dw;
	    if ( tw0>=0 && tw0<=1 ) // we only care if this happens during shutter
		{
	        if ( w0 < 0 )
	            tbounds.x = tw0;
	        else
	            tbounds.y = tw0;
		}
	}

	bool movesright = dx*w0 > dw*x0;
	bool movesforward = dw > 0;

	// compute intercept times with the x bounds,
	// shrink time bounds if possible
	
	//      (x0 + t*dx)/(w0 + t*dw)=xmin
	// <=>  (x0 + t*dx)=xmin*(w0 + t*dw)
	// <=>  t(dx - xmin*dw) = xmin*w0-x0
	
	F32 tmin = (xmin*w0-x0)/(dx-xmin*dw);
	// only care if xmin-intercept happens when w>0
	if ( w0+tmin*dw >= 0 )
	{
	    // does the point enter or exit the screen here?
	    if ( movesright )
	        // point enters screen, cap tmin
	        tbounds.x = max(tbounds.x, tmin );
	    else
	        // point exits screen, cap max
	        tbounds.y = min(tbounds.y, tmin );
	}
	
	// same for xmax
	F32 tmax = (xmax*w0-x0)/(dx-xmax*dw);
	// only care if xmax-intercept happens when w>0
	if ( w0+tmax*dw >= 0 )
	{
	    // does the point enter or exit the screen here?
	    if ( !movesright )
	        // point enters screen, cap tmin
	        tbounds.x = max( tbounds.x, tmax );
	    else
	        // point exits screen, cap max
	        tbounds.y = min( tbounds.y, tmax );
	}

	// the bounds invert if point is not on screen at any time
	if ( tbounds.x > tbounds.y )
	{
	    return false;
	}

	// get the two planes
	// one plane's tangent is the derivative at the t in the middle of bounds
	F32 thalf = 0.5f*(tbounds.x + tbounds.y);
	F32 px = (x0+thalf*dx)/(w0+thalf*dw);
	F32 dxdt = dx/(w0+thalf*dw) - (dw*(x0+thalf*dx))/((w0+thalf*dw)*(w0+thalf*dw));
	
	// evaluate x and ends of time bounds
	F32 xa = (x0+tbounds.x*dx)/(w0+tbounds.x*dw);
	F32 xb = (x0+tbounds.y*dx)/(w0+tbounds.y*dw);
	
	Vec2f plane1( dxdt, (px-dxdt*thalf) );
	Vec2f plane2( -(xb-xa)/(tbounds.x-tbounds.y), xa+((xb-xa)*tbounds.x)/(tbounds.x-tbounds.y) );
	if ( movesright==movesforward )
	{
	    pDest[1] = plane1;
	    pDest[0] = plane2;
	}
	else
	{
	    pDest[0] = plane1;
	    pDest[1] = plane2;
	}

	*pDestSpan = tbounds;

	return true;
}

} //