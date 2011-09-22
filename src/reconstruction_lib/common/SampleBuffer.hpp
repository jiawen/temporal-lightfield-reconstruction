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
#include "base/Array.hpp"
#include "gui/Image.hpp"
#include "base/Random.hpp"

namespace FW
{

//-------------------------------------------------------------------
// Base class provides N samples per pixel. Supports clears and scanout.
//-------------------------------------------------------------------

class CameraParams;
class InterleavedUVTSampleBuffer;
enum ReconstructionFilter;

class SampleBuffer
{
public:

	struct Sample
	{
		Sample()																									{ }
        Sample(int x,int y,int i, const Vec4f& color,float z,float w=0,Vec3f mv=0,Vec2f wg=0) : x(x), y(y), i(i), color(color), z(z), w(w), mv(mv), wg(wg) {}

		S32		x,y,i;	// which sample (x,y, sample number)
		Vec2f	xy;		// not from renderer
		Vec2f	uv;		// not from renderer
		float	t;		// not from renderer
		float	z;		// depth
		float	w;		// camera space z
		Vec4f	color;	// color
		Vec3f	mv;		// motion vector (x,y,w)
		Vec2f	wg;		// dz/dx, dz/dy
	};

					SampleBuffer		(int w,int h, int numSamplesPerPixel);
	virtual			~SampleBuffer		(void)                                  { for(int i=0;i<m_channels.getSize();i++) { delete (Array<int>*)(m_channels[i]); } }

	bool			needRealloc			(int w,int h, int numSamplesPerPixel) const;

	virtual void	clear				(const Vec4f& color,float depth,float w);
	void			scanOut				(Image& image, const CameraParams& params) const;

	int				getWidth			(void) const							{ return m_width; }
	int				getHeight			(void) const							{ return m_height; }
	int				getNumSamples		(void) const							{ return m_numSamplesPerPixel; }
	virtual int		getNumSamples		(int x,int y) const						{ FW_UNREF(x); FW_UNREF(y); return m_numSamplesPerPixel; }	// prepares for variable per-pixel storage

	// i is sample number [0,numSamplesPerPixel)

	const Vec2f&	getSampleXY			(int x,int y, int i) const				{ return m_xy[getIndex(x,y,i)]; }
	void			setSampleXY			(int x,int y,int i, const Vec2f& xy)	{ m_xy[getIndex(x,y,i)] = xy; }

	const Vec4f&	getSampleColor	 	(int x,int y, int i) const				{ return m_color[getIndex(x,y,i)]; }
	void			setSampleColor	 	(int x,int y, int i, const Vec4f& c)	{ m_color[getIndex(x,y,i)]=c; }

	const float&	getSampleDepth	 	(int x,int y, int i) const				{ return m_depth[getIndex(x,y,i)]; }
	void			setSampleDepth	 	(int x,int y, int i, float d)			{ m_depth[getIndex(x,y,i)]=d; }

	const float&	getSampleW			(int x,int y, int i) const				{ return m_w[getIndex(x,y,i)]; }
	void			setSampleW		 	(int x,int y, int i, float d)			{ m_w[getIndex(x,y,i)]=d; }

	const float&    getSampleWeight     (int x,int y, int i) const              { return m_weight[getIndex(x,y,i)]; }
    void            setSampleWeight     (int x,int y,int i, float weight)       { m_weight[getIndex(x,y,i)] = weight; }

	virtual void	setSample			(const Sample& s)						{ setSampleColor(s.x,s.y,s.i,s.color); setSampleDepth(s.x,s.y,s.i,s.z); setSampleW(s.x,s.y,s.i,s.w); }

	// Circle of confusion

	int				validateCircleOfConfusion	(const CameraParams& params, const String& name);
	int				validateCameraSpaceZ		(const CameraParams& params, const String& name);

	// Support for additional per-sample channels.

	float			getSampleFloat		(int id, int x,int y,int i) const		{ return getSampleExtra<float>(id,x,y,i); }
	int				getSampleInt		(int id, int x,int y,int i) const		{ return getSampleExtra<int>(id,x,y,i); }
	void			setSampleFloat		(int id, int x,int y,int i, float val)	{ setSampleExtra<float>(id,x,y,i, val); }
	void			setSampleInt		(int id, int x,int y,int i, int val)	{ setSampleExtra<int>(id,x,y,i, val); }

	template<class T> T    getSampleExtra	(int cid, int x,int y,int i) const		{ if(cid==-1) return T(0); const Array<T>& ec = *(const Array<T>*)(m_channels[cid]); return ec[getIndex(x,y,i)]; }
	template<class T> void setSampleExtra	(int cid, int x,int y,int i, T val)		{ Array<T>& ec = *(Array<T>*)(m_channels[cid]); ec[getIndex(x,y,i)] = val; }
	template<class T> int  reserveChannel	(const String& name)					{ int i = getChannelID(name); if(i==-1){ i = m_channelNames.getSize(); Array<T>*c = new Array<T>; c->reset(m_xy.getSize()); m_channels.add(c); m_channelNames.add(name); } return i; }
	int					   getChannelID		(const String& name) const				{ for(int i=0;i<m_channelNames.getSize();i++) { if(m_channelNames[i]==name) return i; } return -1; }

protected:
	SampleBuffer()	{}
	virtual int		getIndex			(int x,int y,int i) const				{ return (y*m_width+x)*m_numSamplesPerPixel + i; }

	int				m_width;
	int				m_height;
	int				m_numSamplesPerPixel;

	Array<Vec2f>	m_xy;				// for each sample
	Array<Vec4f>	m_color;			// for each sample
	Array<float>	m_depth;			// for each sample
	Array<float>	m_w;				// for each sample
    Array<float>    m_weight;           // scanout weight, for each sample

	Array<void*>	m_channels;
	Array<String>	m_channelNames;

	friend class TreeGather;			// for creating an output sample buffer (DEBUG feature).
};

//-------------------------------------------------------------------
// Adds lens position (uv) and time (t) for each sample
//-------------------------------------------------------------------

class UVTSampleBuffer : public SampleBuffer
{
public:
					UVTSampleBuffer		(int w,int h, int numSamplesPerPixel);
    virtual         ~UVTSampleBuffer    (void)                                  {}

	void			clear				(const Vec4f& color,float depth,float w);
	void			setSample			(const Sample& s)						{ SampleBuffer::setSample(s); setSampleMV(s.x,s.y,s.i,s.mv); setSampleWG(s.x,s.y,s.i,s.wg); }

	float			getSampleT			(int x,int y, int i) const				{ return m_t [getIndex(x,y,i)]; }
	void			setSampleT			(int x,int y,int i, float t)			{ m_t [getIndex(x,y,i)] = t; }

	const Vec2f&	getSampleUV			(int x,int y, int i) const				{ return m_uv[getIndex(x,y,i)]; }
	void			setSampleUV			(int x,int y,int i, const Vec2f& uv)	{ m_uv[getIndex(x,y,i)] = uv; }

	const Vec3f&	getSampleMV			(int x,int y, int i) const				{ return m_mv[getIndex(x,y,i)]; }
	void			setSampleMV			(int x,int y, int i, const Vec3f& mv)	{ m_mv[getIndex(x,y,i)] = mv; }

	const Vec2f&	getSampleWG			(int x,int y, int i) const				{ return m_wg[getIndex(x,y,i)]; }
	void			setSampleWG			(int x,int y, int i, const Vec2f& wg)	{ m_wg[getIndex(x,y,i)] = wg; }

	Vec4f			getXYWFrom			(int x,int y,int i, const Vec2f uv, bool homogeneous) const;	// z=1

	void			setCocCoeffs		(const Vec2f coeff)						{ m_cocCoeff=coeff; }
	const Vec2f&	getCocCoeffs		(void) const							{ return m_cocCoeff; }

	// serialization.

					UVTSampleBuffer			(const char* filename);
	void			serialize				(const char* filename, bool separateHeader=false, bool binary=false) const;

protected:
	UVTSampleBuffer()	{ }

    void            generateSobol       (Random& random);
    void            generateSobolCoop   (Random& random);

	Vec2f			m_cocCoeff;

	Array<Vec2f>	m_uv;			// for each sample
	Array<float>	m_t;			// for each sample
	Array<Vec3f>	m_mv;			// for each sample
	Array<Vec2f>	m_wg;			// for each sample

	friend class TreeGather;		// for creating an output sample buffer (DEBUG feature).
};

} //