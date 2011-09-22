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

#pragma warning(disable:4127)
#pragma warning(disable:4996)

#include "SampleBuffer.hpp"
#include "CameraParams.hpp"
#include "Util.hpp"
#include "base/Sort.hpp"
#include "gui/Image.hpp"
#include "3d/ConvexPolyhedron.hpp"
#include <cstdio>

namespace FW
{

SampleBuffer::SampleBuffer(int w,int h, int numSamplesPerPixel)
{
	m_width  = w;
	m_height = h;
	m_numSamplesPerPixel = numSamplesPerPixel;

	m_color .reset(m_width*m_height*m_numSamplesPerPixel);
	m_depth .reset(m_width*m_height*m_numSamplesPerPixel);
	m_w     .reset(m_width*m_height*m_numSamplesPerPixel);
    m_weight.reset(m_width*m_height*m_numSamplesPerPixel);

	// Generate XY samples

	Random random(242);
	m_xy.reset(m_width*m_height*m_numSamplesPerPixel);
	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	{
		const bool XY_HALTON     = false;	// select one (very minor effect on image quality)
		const bool XY_HAMMERSLEY = false;
		const bool XY_SOBOL	     = true;
		const bool XY_LP		 = false;

		Vec2f offset(random.getF32(),random.getF32());	// [0,1)
		if(m_numSamplesPerPixel<=4)
			offset = 0;		// TODO

		for(int i=0;i<m_numSamplesPerPixel;i++)
		{
			int j=i+1;	// 0 produces 0.0 with all sequences, we don't want that
			Vec2f samplePos;
			if(XY_LP)
			{
				// [0,1) Larcher-Pillichshammer with random scramble.

				samplePos = Vec2f(larcherPillichshammer(j,y*m_width+x),(i+0.5f)/m_numSamplesPerPixel);
			}
			else
			{
				if(XY_HALTON)		samplePos = Vec2f(halton(2,j),halton(3,j));						// [0,1) Halton
				if(XY_HAMMERSLEY)	samplePos = Vec2f(halton(2,j),(i+0.5f)/m_numSamplesPerPixel);	// [0,1) Hammersley
				if(XY_SOBOL)		samplePos = Vec2f(sobol(0,j),sobol(1,j));						// [0,1) Sobol

				// Cranley-Patterson rotations.

				samplePos += offset;
				if(samplePos.x>=1.f) samplePos.x-=1.f;
				if(samplePos.y>=1.f) samplePos.y-=1.f;
			}

			FW_ASSERT(samplePos.x>=0 && samplePos.x<1.f);
			FW_ASSERT(samplePos.y>=0 && samplePos.y<1.f);
			setSampleXY(x,y,i, samplePos + Vec2f(Vec2i(x,y)));
            setSampleWeight(x,y,i, 1.0f);
		}
	}
}

bool SampleBuffer::needRealloc(int w,int h, int numSamplesPerPixel) const
{
	if(w != m_width)								return true;
	if(h != m_height)								return true;
	if(numSamplesPerPixel != m_numSamplesPerPixel)	return true;
	return false;
}

void SampleBuffer::clear(const Vec4f& color,float depth,float w)
{
	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		setSampleColor(x,y,i, color);
		setSampleDepth(x,y,i, depth);
		setSampleW    (x,y,i, w);
	}
}

void UVTSampleBuffer::clear(const Vec4f& color,float depth,float w)
{
	SampleBuffer::clear(color,depth,w);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		setSampleMV	(x,y,i,0);
		setSampleWG (x,y,i,0);
	}
}


void SampleBuffer::scanOut(Image& image, const CameraParams& params) const
{
	const int w = min(image.getSize().x, m_width );
	const int h = min(image.getSize().y, m_height);

	switch(params.filter)
	{
	case FILTER_BOX:
		for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			Vec4f outputColor(0.f);
            float weight = 0.f;
			for(int i=0;i<getNumSamples(x,y);i++)
            {
                float K = getSampleWeight(x,y,i);
				outputColor += K*getSampleColor(x,y,i);
                weight += K;
            }
			image.setVec4f( Vec2i(x,y), outputColor / weight );
		}
		break;

	case FILTER_BICUBIC:
		for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			const Vec2f pixelCenter(x+0.5f,y+0.5f);
			Vec4f outputColor(0.f);
			float weight = 0.f;

			// Neighborhood.

			for(int dy=-2;dy<=2;dy++)
			for(int dx=-2;dx<=2;dx++)
			{
				int nx = x+dx;
				int ny = y+dy;
				if(nx<0 || nx>=w || ny<0 || ny>=h)
					continue;

				for(int i=0;i<getNumSamples(x,y);i++)
				{
					Vec2f samplePos = getSampleXY(nx,ny,i);
					float d = (samplePos-pixelCenter).length();
					if(d>=2.f)
						continue;

					const float B = 1/3.f, C = 1/3.f;	// "optimal" Mitchell-Netravali
					//const float B = 2/3.f, C = -1/4.f;	// "notch" (very blurry)
					float K = (d<=1) ? ((12-9*B-6*C)*d*d*d + (-18+12*B+6*C)*d*d + (6-2*B)) : ((-B-6*C)*d*d*d + (6*B+30*C)*d*d + (-12*B-48*C)*d + (8*B+24*C));
                    K *= getSampleWeight(nx,ny,i);
					outputColor += K*getSampleColor(nx,ny,i);
					weight += K;
				}
			}

			outputColor /= weight;						// normalize
			outputColor = max(outputColor,Vec4f(0.f));	// clamp to (0,1)
			outputColor = min(outputColor,Vec4f(1.f));
			image.setVec4f( Vec2i(x,y), outputColor );
		}
		break;

	default:
		fail("Unknown filter in SampleBuffer::scanOut()");
	}
}

//-------------------------------------------------------------------

UVTSampleBuffer::UVTSampleBuffer(int w,int h, int numSamplesPerPixel)
: SampleBuffer(w,h,numSamplesPerPixel)
{
	Random random(1);

	m_cocCoeff     = Vec2f(FW_F32_MAX,FW_F32_MAX);

	m_uv.reset(m_width*m_height*m_numSamplesPerPixel);
	m_t. reset(m_width*m_height*m_numSamplesPerPixel);
	m_mv.reset(m_width*m_height*m_numSamplesPerPixel);
	m_wg.reset(m_width*m_height*m_numSamplesPerPixel);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width ;x++)
	for(int i=0;i<m_numSamplesPerPixel;i++)
	{
		m_mv[ getIndex(x,y,i) ] = Vec3f(0,0,0);
		m_wg[ getIndex(x,y,i) ] = Vec2f(0,0);
	}

//  generateSobol(random);
    generateSobolCoop(random);
}


UVTSampleBuffer::UVTSampleBuffer(const char* filename)
{
	FILE* fp  = fopen(filename, "rb");
	if(!fp)
		fail("File not found");
	FILE* fph = fopen((String(filename)+String(".header")).getPtr(),"rt");
	bool separateHeader = (fph!=NULL);
	if(!separateHeader)
		fph = fp;

	printf("Importing sample buffer... ");

	// Parse version and image size.

	float version;
	fscanf(fph, "Version %f\n", &version);
	fscanf(fph, "Width %d\n", &m_width);
	fscanf(fph, "Height %d\n", &m_height);
	fscanf(fph, "Samples per pixel %d\n", &m_numSamplesPerPixel);

	if(version == 1.3f)
	{
		// Parse the rest of the header.

		char motionModel[1024];
		fscanf(fph, "Motion model: %s\n", motionModel);
		//m_affineMotion = (String(motionModel) == String("affine"));	// deprecated

		fscanf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", &m_cocCoeff[0],&m_cocCoeff[1]);

		bool binary = false;
		char encoding[1024];
		if(fscanf(fph, "Encoding = %s\n", encoding)==1)
			binary = String(encoding) == String("binary");

		fscanf(fph, "\n");

		char descriptor[1024];
		fscanf(fph, "%s\n", descriptor);

		// Reserve buffers.

		m_xy   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_uv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_t    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_color.reset(m_width*m_height*m_numSamplesPerPixel);
		m_depth.reset(m_width*m_height*m_numSamplesPerPixel);
		m_w    .reset(m_width*m_height*m_numSamplesPerPixel);
		m_mv   .reset(m_width*m_height*m_numSamplesPerPixel);
		m_wg   .reset(m_width*m_height*m_numSamplesPerPixel);

		// Parse samples.

		struct Entry
		{
			float x,y,z,w,u,v,t,r,g,b,a,mv_x,mv_y,mv_w,dwdx,dwdy;
		};

		printf("\n");
		if(!binary)
		{
			char line[4096];
			for(int y=0;y<m_height;y++)
			{
				for(int x=0;x<m_width;x++)
				for(int i=0;i<getNumSamples(x,y);i++)
				{
					fgets(line,4096,fp);
					const char* linePtr = line;
					const int NUM_ARGS = sizeof(Entry)/sizeof(float);
					float vals[NUM_ARGS];
					int numRead = 0;
					for(int v=0;v<NUM_ARGS;v++)
					{
						if(v>0)
						{
							while(*linePtr!=',' && *linePtr!='\n')
								linePtr++;
							linePtr++;	// skip ','
						}

						parseFloat(linePtr,vals[v]);
						numRead++;
					}

					Entry e;
					e.x    = vals[0];
					e.y    = vals[1];
					e.z    = vals[2];
					e.w    = vals[3];
					e.u    = vals[4];
					e.v    = vals[5];
					e.t    = vals[6];
					e.r    = vals[7];
					e.g    = vals[8];
					e.b    = vals[9];
					e.a    = vals[10];
					e.mv_x = vals[11];
					e.mv_y = vals[12];
					e.mv_w = vals[13];
					e.dwdx = vals[14];
					e.dwdy = vals[15];

					FW_ASSERT(numRead == NUM_ARGS);
					FW_ASSERT(e.x>=0 && e.y>=0 && e.x<m_width && e.y<m_height);
					FW_ASSERT(e.u>=-1 && e.v>=-1 && e.u<=1 && e.v<=1);
					FW_ASSERT(e.t>=0 && e.t<=1);

					setSampleXY				(x,y,i, Vec2f(e.x,e.y));
					setSampleDepth			(x,y,i, e.z);
					setSampleW				(x,y,i, e.w);
					setSampleUV				(x,y,i, Vec2f(e.u,e.v));
					setSampleT 				(x,y,i, e.t);
					setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));			// TODO: alpha
					setSampleMV				(x,y,i, Vec3f(e.mv_x,e.mv_y,e.mv_w));
					setSampleWG				(x,y,i, Vec2f(e.dwdx,e.dwdy));
				}

				printf("%d%%\r", 100*y/m_height);
			}
		}
		else
		{
			Array<Entry> entries;
			const int num = m_width*m_height*m_numSamplesPerPixel;
			entries.reset(num);

			fread(entries.getPtr(),sizeof(Entry),num,fp);

			int sidx = 0;
			for(int y=0;y<m_height;y++)
			{
				for(int x=0;x<m_width;x++)
				for(int i=0;i<getNumSamples(x,y);i++)
				{
					const Entry& e = entries[sidx++];

					setSampleXY				(x,y,i, Vec2f(e.x,e.y));
					setSampleDepth			(x,y,i, e.z);
					setSampleW				(x,y,i, e.w);
					setSampleUV				(x,y,i, Vec2f(e.u,e.v));
					setSampleT 				(x,y,i, e.t);
					setSampleColor			(x,y,i, Vec4f(e.r,e.g,e.b,1));			// TODO: alpha
					setSampleMV				(x,y,i, Vec3f(e.mv_x,e.mv_y,e.mv_w));
					setSampleWG				(x,y,i, Vec2f(e.dwdx,e.dwdy));
				}
				printf("%d%%\r", 100*y/m_height);
			}
		}
	}
	else
		fail("Unsupported sample stream version (%.1f)", version);

	fclose(fp);
	if(separateHeader)
		fclose(fph);
	printf("done\n");
}

Vec4f UVTSampleBuffer::getXYWFrom(int x,int y,int i, const Vec2f uv, bool homogeneous) const
{
	// NOTE: No longer reprojects t. 

	const float w  = getSampleW(x,y,i);
	const Vec2f p  = getSampleXY(x,y,i) - getCocRadius(m_cocCoeff,w)*getSampleUV(x,y,i);	// uv=0
	Vec4f xyw(p+getCocRadius(m_cocCoeff,w)*uv, 1, w);										// affine position @ output (u,v)
	if(homogeneous)
	{
		xyw.x *= xyw.w;
		xyw.y *= xyw.w;
	}
	return xyw;
}

void UVTSampleBuffer::serialize(const char* filename, bool separateHeader, bool binary) const
{
	if(binary && !separateHeader)
		fail("binary serialization supported only with a separate header");

	FILE* fp = (binary) ? fopen(filename, "wb") : fopen(filename, "wt");
	FILE* fph= (separateHeader) ? fopen((String(filename)+String(".header")).getPtr(),"wt") : fp;

	printf("Serializing sample buffer... ");

	const float version = 1.3f;

	if(version==1.2f)
	{
		// header
		fprintf(fph, "Version 1.2\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		fprintf(fph, "\n");
		fprintf(fph, "x,y,u,v,t,r,g,b,z,coc_radius,motion_x,motion_y,wgrad_x,wgrad_y\n");

		const int CID = getChannelID("COC");

		// samples
		for(int y=0;y<m_height;y++)
		for(int x=0;x<m_width;x++)
		for(int i=0;i<getNumSamples(x,y);i++)
		{
			fprintf(fp, "%f,", getSampleXY(x,y,i)[0]);			// x
			fprintf(fp, "%f,", getSampleXY(x,y,i)[1]);			// y
			fprintf(fp, "%f,", getSampleUV(x,y,i)[0]);			// u
			fprintf(fp, "%f,", getSampleUV(x,y,i)[1]);			// v
			fprintf(fp, "%f,", getSampleT (x,y,i));				// t

			fprintf(fp, "%f,", getSampleColor(x,y,i)[0]);		// r
			fprintf(fp, "%f,", getSampleColor(x,y,i)[1]);		// g
			fprintf(fp, "%f,", getSampleColor(x,y,i)[2]);		// b
			fprintf(fp, "%f,", getSampleDepth(x,y,i));			// z

			fprintf(fp, "%f,", getSampleFloat(CID,x,y,i));		// coc radius
			fprintf(fp, "%f,", getSampleMV(x,y,i)[0]);			// motion vector.x
			fprintf(fp, "%f,", getSampleMV(x,y,i)[1]);			// motion vector.y
			fprintf(fp, "%f,", getSampleWG(x,y,i)[0]);			// w gradient.x
			fprintf(fp, "%f",  getSampleWG(x,y,i)[1]);			// w gradient.y
			fprintf(fp, "\n"); 
		}
	} // 1.2

	else if(version==1.3f)
	{
		if(m_cocCoeff == Vec2f(FW_F32_MAX,FW_F32_MAX))
			fail("coc coefficients not set");

		// header
		fprintf(fph, "Version 1.3\n");
		fprintf(fph, "Width %d\n", m_width);
		fprintf(fph, "Height %d\n", m_height);
		fprintf(fph, "Samples per pixel %d\n", m_numSamplesPerPixel);
		//fprintf(fph, "Motion model: %s\n", m_affineMotion ? "affine" : "perspective");	// deprecated
		fprintf(fph, "CoC coefficients (coc radius = C0/w+C1): %f,%f\n", m_cocCoeff[0], m_cocCoeff[1]);
		fprintf(fph, "Encoding = %s\n", binary ? "binary" : "text");
		fprintf(fph, "x,y,z/w,w,u,v,t,r,g,b,a,mv_x,mv_y,mv_w,dwdx,dwdy\n");

		// Using Wikipedia's terminology
		//
		//    c = AD * (objectDist-focusDist)/objectDist * f/(focusDist-f)
		// => c = AD * f/(focusDist-f) * (1-focusDist/objectDist)
		// => c = C1 * (1-focusDist/objectDist)
		// => c = C1 - C1*focusDist/objectDist
		// => c = C1 + C0/objectDist
		//
		// C1 = ApertureDiameter * f/(focusDist-f)
		// C0 = -C1*focusDist

		if(!binary)
		{
			// samples
			for(int y=0;y<m_height;y++)
			for(int x=0;x<m_width;x++)
			for(int i=0;i<getNumSamples(x,y);i++)
			{
				fprintf(fp, "%f,", getSampleXY(x,y,i)[0]);			// x	(in window coordinates, NOT multiplied with w)
				fprintf(fp, "%f,", getSampleXY(x,y,i)[1]);			// y	(in window coordinates, NOT multiplied with w)
				fprintf(fp, "%f,", getSampleDepth(x,y,i));			// z	(z/w as in OpenGL, not used by reconstruction)
				fprintf(fp, "%f,", getSampleW(x,y,i));				// w	(camera-space z. Positive are visible, larger is farther).

				fprintf(fp, "%f,", getSampleUV(x,y,i)[0]);			// u	[-1,1]
				fprintf(fp, "%f,", getSampleUV(x,y,i)[1]);			// v	[-1,1]
				fprintf(fp, "%f,", getSampleT (x,y,i));				// t	[0,1]

				fprintf(fp, "%f,", getSampleColor(x,y,i)[0]);		// r	[0,1]
				fprintf(fp, "%f,", getSampleColor(x,y,i)[1]);		// g	[0,1]
				fprintf(fp, "%f,", getSampleColor(x,y,i)[2]);		// b	[0,1]
				fprintf(fp, "%f,", getSampleColor(x,y,i)[2]);		// a	[0,1]				TODO

				fprintf(fp, "%f,", getSampleMV(x,y,i)[0]);			// homogeneous motion vector.x
				fprintf(fp, "%f,", getSampleMV(x,y,i)[1]);			// homogeneous motion vector.y
				fprintf(fp, "%f,", getSampleMV(x,y,i)[2]);			// homogeneous motion vector.w

				fprintf(fp, "%f,", getSampleWG(x,y,i)[0]);			// dw/dx
				fprintf(fp, "%f,", getSampleWG(x,y,i)[1]);			// dw/dy
				fprintf(fp, "\n"); 
			}
		}
		else
		{
			struct Entry
			{
				float x,y,z,w,u,v,t,r,g,b,a,mv_x,mv_y,mv_w,dwdx,dwdy;
			};

			int num = 0;
			for(int y=0;y<m_height;y++)
			for(int x=0;x<m_width; x++)
				num += getNumSamples(x,y);

			Array<Entry> entries;
			entries.reset(num);

			int sidx=0;
			for(int y=0;y<m_height;y++)
			for(int x=0;x<m_width;x++)
			for(int i=0;i<getNumSamples(x,y);i++)
			{
				Entry& e = entries[sidx++];
				e.x = getSampleXY(x,y,i)[0];			// x	(in window coordinates, NOT multiplied with w)
				e.y = getSampleXY(x,y,i)[1];			// y	(in window coordinates, NOT multiplied with w)
				e.z = getSampleDepth(x,y,i);			// z	(z/w as in OpenGL, not used by reconstruction)
				e.w = getSampleW(x,y,i);				// w	(camera-space z. Positive are visible, larger is farther).

				e.u = getSampleUV(x,y,i)[0];			// u	[-1,1]
				e.v = getSampleUV(x,y,i)[1];			// v	[-1,1]
				e.t = getSampleT (x,y,i);				// t	[0,1]

				e.r = getSampleColor(x,y,i)[0];			// r	[0,1]
				e.g = getSampleColor(x,y,i)[1];			// g	[0,1]
				e.b = getSampleColor(x,y,i)[2];			// b	[0,1]
				e.a = getSampleColor(x,y,i)[2];			// a	[0,1]				TODO

				e.mv_x = getSampleMV(x,y,i)[0];			// homogeneous motion vector.x
				e.mv_y = getSampleMV(x,y,i)[1];			// homogeneous motion vector.y
				e.mv_w = getSampleMV(x,y,i)[2];			// homogeneous motion vector.w

				e.dwdx = getSampleWG(x,y,i)[0];			// dw/dx
				e.dwdy = getSampleWG(x,y,i)[1];			// dw/dy
			}

			fwrite(entries.getPtr(), sizeof(Entry), num, fp);
		}
	} // 1.3

	fflush(fp);
	fclose(fp);
	if(separateHeader)
	{
		fflush(fph);
		fclose(fph);
	}
	printf("done\n");
}

//-------------------------------------------------------------------

void UVTSampleBuffer::generateSobol(Random& random)
{
    for(int y=0;y<m_height;y++)
    for(int x=0;x<m_width ;x++)
    {
        Vec3f offset = random.getVec3f();
        for(int i=0;i<m_numSamplesPerPixel;i++)
        {
            float u = hammersley(i, m_numSamplesPerPixel);
            Vec2f vt = sobol2D(i);

            u += offset.x, vt.x += offset.y, vt.y += offset.z;
            u -= floor(u), vt.x -= floor(vt.x), vt.y -= floor(vt.y);

            m_uv[(y*m_width+x)*m_numSamplesPerPixel + i] = ToUnitDisk(Vec2f(u, vt.x));
            m_t [(y*m_width+x)*m_numSamplesPerPixel + i] = vt.y;
        }
    }
}

//-------------------------------------------------------------------

void UVTSampleBuffer::generateSobolCoop(Random& random)
{
    Array<Vec4i> shuffle;
    shuffle.reset(24151); // prime
    for (int i = 0; i < shuffle.getSize(); i++)
    {
        shuffle[i] = Vec4i(0, 1, 2, 3);
        for (int j = 4; j >= 2; j--)
            swap(shuffle[i][j - 1], shuffle[i][random.getS32(j)]);
    }

    int sampleIdx = 0;
    for (int py = 0; py < m_height; py++)
    for (int px = 0; px < m_width; px++)
    {
        int morton = 0;
        for (int i = 10; i >= 0; i--)
        {
            int childIdx = ((px >> i) & 1) + ((py >> i) & 1) * 2;
            morton = morton * 4 + shuffle[morton % shuffle.getSize()][childIdx];
        }

        for (int i = 0; i < m_numSamplesPerPixel; i++)
        {
            int j = i + morton * m_numSamplesPerPixel;
            float x = sobol(3, j);
            float y = sobol(4, j);
            float u = sobol(0, j);
            float v = sobol(2, j);
            float t = sobol(1, j);

            m_xy[sampleIdx] = Vec2f(px + x, py + y);
            m_uv[sampleIdx] = ToUnitDisk(Vec2f(u, v));
            m_t[sampleIdx] = t;
            sampleIdx++;
        }
    }
}

//-------------------------------------------------------------------

int SampleBuffer::validateCircleOfConfusion(const CameraParams& params, const String& name)
{
	const int CID = reserveChannel<float>(name);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width; x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		const float z = getSampleDepth(x,y,i);
		float coc = params.getCocRadius(z);
		setSampleFloat(CID, x,y,i, coc);
	}

	return CID;
}

int SampleBuffer::validateCameraSpaceZ		(const CameraParams& params, const String& name)
{
	const int CID = reserveChannel<float>(name);

	for(int y=0;y<m_height;y++)
	for(int x=0;x<m_width; x++)
	for(int i=0;i<getNumSamples(x,y);i++)
	{
		const float z = getSampleDepth(x,y,i);
		float coc = params.getCameraSpaceZ(z);
		setSampleFloat(CID, x,y,i, coc);
	}

	return CID;
}

} //