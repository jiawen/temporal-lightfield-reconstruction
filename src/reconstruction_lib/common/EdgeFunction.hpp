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

namespace FW
{

//-----------------------------------------------------------------------------
// Standard (2D) edge function
//-----------------------------------------------------------------------------

class EdgeFunction
{
public:
			EdgeFunction()	{ }
			EdgeFunction(F64 x0, F64 y0, F64 x1, F64 y1);							// traditional
		    EdgeFunction(F64 x0w0, F64 y0w0, F64 w0, F64 x1w1, F64 y1w1, F64 w1);	// homogeneous

	bool	test		(float dist) const			{ return dist>0.f ? true : (dist==0.f && m_acceptEqual); }
	bool	test		(F64 x, F64 y) const		{ return test(evaluate(x,y)); }
	float	evaluate	(F64 x, F64 y) const		{ return (F32)(x*m_A + y*m_B + m_C); }
    Vec2f   gradient    (void) const                { return Vec2f((F32)m_A, (F32)m_B); }

private:
	F64     m_A;
	F64     m_B;
	F64     m_C;
	bool	m_acceptEqual;
};


} //


