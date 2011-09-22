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
#include "3d/CameraControls.hpp"
#include "SampleBuffer.hpp"

namespace FW
{

enum ReconstructionMode
{
	// This is the only available reconstruction mode.
	// The "2" is due to historical reasons, and we chose
	// not to change it to maintain compatibility with
	// the internal code branch.
	// This enum makes it simpler to add your own methods.
	RECONSTRUCTION_TRIANGLE2,
};

enum LensFilter
{
	LENS_BOX,		// default
	LENS_GAUSSIAN
};

enum TimeFilter
{
	TIME_BOX,		// default
	TIME_GAUSSIAN
};

enum AXIS
{
	AXIS_X = 0,
	AXIS_Y = 1,
	AXIS_Z = 2,
};

enum ReconstructionFilter	// for scanout
{
	FILTER_BOX	   = 0,
	FILTER_BICUBIC = 1,
};

// NOTE: cameraSpaceZ>0 means visible
inline float getCocRadius (const Vec2f& coeffs, float cameraSpaceZ)	{ return coeffs[0]/cameraSpaceZ + coeffs[1]; };

class CameraParams
{
public:
	CameraParams(CommonControls* commonControls = NULL, U32 features = CameraControls::Feature_Default) : 
		reconstruction			(RECONSTRUCTION_TRIANGLE2),
		filter					(FILTER_BOX),
		lensFilter				(LENS_BOX),
		timeFilter				(TIME_BOX),
		camera					(commonControls, features),
		windowSize				(0),
		apertureRadius			(0),
		focusDistance			(1),
		overrideUVT				(FW_F32_MAX),
		overrideRefocusDistance	(FW_F32_MAX),
		enableCuda				(false)
	{
	}

	ReconstructionMode		reconstruction;			// which reconstruction to use?
	ReconstructionFilter	filter;					// for scanout/resolve
	LensFilter				lensFilter;				
	TimeFilter				timeFilter;
	CameraControls			camera;					// OpenGL camera, essentially
	Mat4f					projection;				// includes aspect ratio to match Tero's OpenGL rendering

	Vec2i					windowSize;				

	float					apertureRadius;			// 0 == pinhole
	float					focusDistance;			// 

	Vec3f					overrideUVT;			// for producing movies
	float					overrideRefocusDistance;

	bool					enableCuda;

	Mat4f	getWindowScale		(void) const	{ return Mat4f::scale(Vec3f(0.5f*windowSize.x, 0.5f*windowSize.y, 0.5f)) * Mat4f::translate(Vec3f(1.0f)); }	// [-1,1] -> [window size]
	Mat4f	getInvWindowScale	(void) const	{ return Mat4f::translate(Vec3f(-1.0f)) * Mat4f::scale(Vec3f(2.f/windowSize.x, 2.f/windowSize.y, 2.f)); }	// [window size] -> [-1,1]

	inline Vec2f getCocCoeffs(void)	const										{ Mat4f m=getCameraToCocRadius(); return Vec2f(-m.m03, m.m02)*float(windowSize.x); }
	inline float getCocRadius(float depthBufferValue) const						{ return FW::getCocRadius(getCocCoeffs(), getCameraSpaceZ(depthBufferValue)); }



	Mat4f	getDofMatrix(Vec2f uv) const
	{
		Mat4f dof;
		dof.col(2) = Vec4f(uv * apertureRadius/focusDistance, 1, 0);
		dof.col(3) = Vec4f(uv * apertureRadius, 0, 1);
		return dof;
	}

	Mat4f	getCameraToCocRadius(void) const
	{
		Mat4f maxDof = getDofMatrix(Vec2f(1,0) / 2);	// maximum extent (radius) of the lens.
		// Division by 2 transforms the result from [-1,1] -> [0,1]. We don't need translation here
		// because the result is a direction, not position.
		Mat4f CameraToCoc = (projection * maxDof - projection);
		return CameraToCoc;
	}

	inline float getCameraSpaceZ(float depthBufferValue) const
	{
		// NOTE: visible values are positive.
		// Works also for parallel projection. For perspective projection m33=0.
		float t = 2*depthBufferValue-1;	// [-1,1]
		return -(projection.m23 - t*projection.m33) / (t*projection.m32 - projection.m22);
	}
};


} //