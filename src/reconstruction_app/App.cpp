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

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4127)

#include "App.hpp"
#include "base/Main.hpp"
#include "gpu/GLContext.hpp"
#include "3d/Mesh.hpp"
#include "io/Stream.hpp"
#include "io/StateDump.hpp"
#include "io/AviExporter.hpp"

#include <stdio.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

inline float smoothStepEase(float t, float ease)
{
	if (ease < 1e-5f)
		return t;
	float d = (1.f-ease) * 0.5f;
	float e = 0.5f-d;
	float h = d*3.f+2.f*e;
	t = min(max(t, 0.f), 1.f);
	if (t <= e)
	{
		t *= 0.5f/e;
		t = t*t*(3.f-2.f*t);
		t *= e/0.5f;
	} else if (t >= 1.f-e)
	{
		t = 1.f-t;
		t *= 0.5f/e;
		t = t*t*(3.f-2.f*t);
		t *= e/0.5f;
		t = h-t;
	} else
	{
		t -= e;
		t /= (2.f*d);
		t = e + 3.f*d*t;
	}
	t /= h;
	return t;
}

App::App(void)
:   m_commonCtrl    					(CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
	m_cameraParams  					(&m_commonCtrl, 0),
    m_action        					(Action_None),
	m_samples							(NULL),
	m_inputImage						(NULL),
	m_groundTruthImage					(NULL),
	m_reconstructionImage				(NULL),
	m_reconstructionImageCuda			(NULL),
	m_groundTruthPinholeImage			(NULL),
	m_reconstructionPinholeImage		(NULL),
	m_reconstructionPinholeImageCuda	(NULL),
	m_debugImage						(NULL),
	m_imageC							(NULL),
	m_pinholeImageT						(NULL),
	m_pinholeImageC						(NULL),
	m_debugImageT						(NULL),
	m_haveSampleBuffer					(false),
	m_vizDone							(0),
	m_vizDoneCuda						(0),
	m_flipY								(true),
	m_gamma								(1.6f),
	m_focalDistance						(1.f),
	m_showImage							(0),
	m_displayNoCuda						(false)
{
    m_commonCtrl.showFPS(false);
    m_commonCtrl.addStateObject(this);
	m_cameraParams.camera.setKeepAligned(true);

	m_commonCtrl.setStateFilePrefix( "state_reconstruction_app_" );

    m_commonCtrl.addButton((S32*)&m_action, Action_LoadSampleBuffer,    FW_KEY_L,       "Load sample buffer... [L]");
    m_commonCtrl.addButton((S32*)&m_action, Action_SaveSampleBuffer,    FW_KEY_S,       "Save sample buffer... [S]");
	m_commonCtrl.addToggle(&m_flipY,									FW_KEY_Y,		"Flip Y [Y]");
#if (FW_USE_CUDA)
	m_commonCtrl.addToggle(&m_cameraParams.enableCuda,                  FW_KEY_SPACE,	"Enable CUDA [SPACE]");
#else
	m_commonCtrl.addToggle(&m_displayNoCuda,							FW_KEY_SPACE,	"Enable CUDA [SPACE]");
#endif
	m_commonCtrl.addButton((S32*)&m_action, Action_ClearImages,			FW_KEY_DELETE,	"Invalidate all images [DELETE]");
	m_window.addListener(&m_cameraParams.camera);

    m_commonCtrl.addSeparator();

    m_commonCtrl.addButton((S32*)&m_action, Action_ExportUVSweep,       FW_KEY_U,       "Export uv sweep avi... [U]");
	m_commonCtrl.addButton((S32*)&m_action, Action_RunSweep,			FW_KEY_R,		"Run refocus sweep [R]");

    m_commonCtrl.addSeparator();

	m_commonCtrl.addToggle(&m_showImage, VIZ_INPUT						, FW_KEY_F1, 	"Show input (box filter) [F1]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_GROUNDTRUTH				, FW_KEY_F2, 	"Show ground truth [F2]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION				, FW_KEY_F3, 	"Show 128spp reconstruction [F3]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_GROUNDTRUTH_PINHOLE		, FW_KEY_F4, 	"Show ground truth (pinhole) [F4]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_RECONSTRUCTION_PINHOLE		, FW_KEY_F5, 	"Show pinhole reconstruction [F5]");
	m_commonCtrl.addToggle(&m_showImage, VIZ_NUM_SURFACES				, FW_KEY_F6, 	"Show #surfaces [F6]");

    m_window.setTitle("Temporal lightfield reconstruction");
	m_window.setSize(Vec2i(640,480));
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

	// sliders for camera parameters
	m_commonCtrl.beginSliderStack();
	m_commonCtrl.addSlider(&m_focalDistance, 0.01f, 100.0f, true, FW_KEY_NONE,FW_KEY_NONE, "Focal distance *= %.2f", 0.1f);
	m_commonCtrl.addSlider(&m_gamma, 1.f,2.5f,false, FW_KEY_NONE,FW_KEY_NONE, "Gamma %.2f", 0.1f);
	m_commonCtrl.endSliderStack();

	// load state
    m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1));

	m_commonCtrl.flashButtonTitles();
}

//------------------------------------------------------------------------

App::~App(void)
{
	delete m_samples;
	delete m_inputImage;
	delete m_groundTruthImage;
	delete m_reconstructionImage;
	delete m_reconstructionImageCuda;
	delete m_groundTruthPinholeImage;
	delete m_reconstructionPinholeImage;
	delete m_reconstructionPinholeImageCuda;
	delete m_debugImage;
	delete m_imageC;
	delete m_pinholeImageT;
	delete m_pinholeImageC;
	delete m_debugImageT;
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
	if (ev.type == Window::EventType_Close)
    {
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }

    Action action = m_action;
    m_action = Action_None;
    String name;
    Mat4f mat;

    switch (action)
    {
    case Action_None:
        break;

	case Action_LoadSampleBuffer:
		name = m_window.showFileLoadDialog("Load sample buffer","txt:ASCII Sample Buffer,bin:Binary Sample Buffer", "data");
        if (name.getLength())
            importSampleBuffer(name);
        break;

	case Action_SaveSampleBuffer:
        name = m_window.showFileSaveDialog("Save sample buffer");
        if (name.getLength())
			m_samples->serialize(name.getPtr(), true, true);
        break;

	case Action_ExportUVSweep:
		if(!m_samples)
		{
	        m_commonCtrl.message("Sample buffer not imported!");
		}
		else
		{
			name = m_window.showFileSaveDialog("Save uv sweep avi");
			if (name.getLength())
				exportAVI(name);
		}
        break;

	case Action_ClearImages:
		m_vizDone = 0;
		m_vizDoneCuda = 0;
		break;

	case Action_RunSweep:
		{
			float aper0  = 1.f;
			float aper1  = 1.f;
			float focus0 = 2.f;
			float focus1 = 0.1f;
			int   fps    = 25;		// NOTE: premiere offers only a few choices for this
			int   frames = 200;		// "good" frames, don't include pre/post static
			float ease   = 0.70f;
			float btime0 = 0.20f;
			float btime1 = 0.20f;

			m_gamma = 1.6f;
			Image* image = new Image(m_window.getSize(), ImageFormat::RGBA_Vec4f);

			String sweepName = "sweep_";
			SYSTEMTIME st;
			FILETIME ft;
			GetSystemTime(&st);
			SystemTimeToFileTime(&st, &ft);
			U64 stamp = *(const U64*)&ft;
			for (int i = 60; i >= 0; i -= 4)
				sweepName += (char)('a' + ((stamp >> i) & 15));
			AviExporter avi(sweepName + ".avi", m_window.getSize(), fps);
			for (int i=0; i < frames; i++)
			{
				FW::printf("\n** FRAME %d / %d **\n\n", i, frames);

				float t = smoothStepEase((float)i / (frames-1), ease);
				float f = 1.f/(1.f/focus0 + t*(1.f/focus1 - 1.f/focus0));
				float a = aper0 + t*(aper1-aper0);

				m_cameraParams.reconstruction = RECONSTRUCTION_TRIANGLE2;
				TreeGather filter(*m_samples, m_cameraParams, a, f);
				filter.reconstructDofMotion(*image);
				adjustGamma(*image);
				exportImage(sweepName + sprintf("_frame%03d.png", i), image);
				avi.getFrame() = *image;
				int n = 1;
				if (i==0) n = (int)(btime0*fps+.5f);
				if (i==frames-1) n = (int)(btime1*fps+.5f);
				while (n--)
					avi.exportFrame();
				blitToWindow(m_window.getGL(), *image);
			    m_window.getGL()->swapBuffers();
			}			
			avi.flush();
			FW::printf("Sweep done\n");

			delete image;
		}
		// sweep parameters
		break;

    default:
        FW_ASSERT(false);
        break;
    }

    m_window.setVisible(true);

    if (ev.type == Window::EventType_Paint)
        render(m_window.getGL());
    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& d)
{
	printf("readState\n");

    d.pushOwner("App");
    String fileName;
	d.get(fileName, "m_fileName");
    d.get((S32&)m_flipY, "m_flipY");
    d.get((bool&)m_cameraParams.enableCuda, "enableCuda");
    d.get((F32&)m_gamma, "m_gamma");
    d.popOwner();

	if(fileName.getLength())
		importSampleBuffer(fileName);
}

//------------------------------------------------------------------------

void App::writeState(StateDump& d) const
{
	printf("writeState\n");

    d.pushOwner("App");
	d.set(m_fileName, "m_fileName");
    d.set((S32&)m_flipY, "m_flipY");
    d.set((bool&)m_cameraParams.enableCuda, "enableCuda");
    d.set((F32&)m_gamma, "m_gamma");
    d.popOwner();
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
}

//------------------------------------------------------------------------

void FW::init(void)
{
	new App;
}

//------------------------------------------------------------------------

void App::importSampleBuffer (const String& fileName)
{
	m_fileName         = fileName;
	m_haveSampleBuffer = true;
	m_vizDone		   = 0;
	m_vizDoneCuda	   = 0;

	// Import sample buffer.

	delete m_samples;
	m_samples = new UVTSampleBuffer(fileName.getPtr());

	// Resize window and image.

	const Vec2i windowSize( m_samples->getWidth(),m_samples->getHeight() );
	m_window.setSize( windowSize );

	delete m_inputImage;
	delete m_groundTruthImage;
	delete m_reconstructionImage;
	delete m_reconstructionImageCuda;
	delete m_groundTruthPinholeImage;
	delete m_reconstructionPinholeImage;
	delete m_reconstructionPinholeImageCuda;
	delete m_debugImage;
	delete m_imageC;
	delete m_pinholeImageT;
	delete m_pinholeImageC;
	delete m_debugImageT;

	m_inputImage						= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_reconstructionImage				= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_reconstructionImageCuda			= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_reconstructionPinholeImage  		= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_reconstructionPinholeImageCuda 	= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_debugImage						= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_imageC		 					= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_pinholeImageT  					= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_pinholeImageC  					= new Image(windowSize, ImageFormat::RGBA_Vec4f);
	m_debugImageT						= new Image(windowSize, ImageFormat::RGBA_Vec4f);

	m_inputImage->clear();
    m_reconstructionImage->clear();
	m_reconstructionImageCuda->clear();
    m_reconstructionPinholeImage->clear();
    m_reconstructionPinholeImageCuda->clear();
    m_debugImage->clear();
	m_imageC->clear();
	m_pinholeImageT->clear();
    m_pinholeImageC->clear();
    m_debugImageT->clear();

	// compute input image by bucketing the input samples (box filter)
	for(int y=0;y<m_samples->getHeight();y++)
	for(int x=0;x<m_samples->getWidth();x++)
	for(int i=0;i<m_samples->getNumSamples(x,y);i++)
	{
		Vec2f p = m_samples->getSampleXY(x,y,i);
		Vec2i pi((int)floor(p.x),(int)floor(p.y));
		if(pi.x<0 || pi.y<0 || pi.x>=m_inputImage->getSize().x || pi.y>=m_inputImage->getSize().y)
			continue;
		Vec4f c = m_samples->getSampleColor(x,y,i);
		m_inputImage->setVec4f(pi, m_inputImage->getVec4f(pi)+c);
	}
	for(int y=0;y<m_inputImage->getSize().y;y++)
	for(int x=0;x<m_inputImage->getSize().x;x++)
	{
		Vec4f c = m_inputImage->getVec4f(Vec2i(x,y));
		c *= rcp(c.w);
		m_inputImage->setVec4f(Vec2i(x,y), c);
	}

	// TODO fix png

	// try loading ground truth images
	WIN32_FIND_DATA FF;
	String gtf = fileName + ".groundtruth.?.?.png";
	HANDLE hFF = FindFirstFile( gtf.getPtr(), &FF );
	U32 gammaI = 0, gammaF = 0;
	if ( hFF != INVALID_HANDLE_VALUE )
	{
		String ending = String(FF.cFileName).substring( (int)(strstr( FF.cFileName, "groundtruth" )-FF.cFileName) );
		int tokens = sscanf_s( ending.getPtr(), "groundtruth.%d.%d.png", &gammaI, &gammaF );
		if ( tokens != 2 )
			fail( "Invalid format for ground truth image file (%s)", FF.cFileName );
		m_groundTruthImage = importImage( FW::sprintf( "%s.groundtruth.%d.%d.png", fileName.getPtr(), gammaI, gammaF ) );
		m_gamma = gammaI + gammaF/10.0f;

		FindClose( hFF );
		gtf = FW::sprintf( "%s.groundtruth.%d.%d.pinhole.png", fileName.getPtr(), gammaI, gammaF );
		m_groundTruthPinholeImage = importImage( gtf );
	}
	else
	{
		m_groundTruthImage = new Image(windowSize, ImageFormat::RGBA_Vec4f);
		m_groundTruthImage->clear();
		m_groundTruthPinholeImage = new Image(windowSize, ImageFormat::RGBA_Vec4f);
		m_groundTruthPinholeImage->clear();
	}
}

//------------------------------------------------------------------------

void App::reconstructPinhole(Visualization viz)
{
	U32& vizDone = (m_cameraParams.enableCuda ? m_vizDoneCuda : m_vizDone);
	if(!m_haveSampleBuffer || (vizDone&(1<<viz)))
		return;

	printf("\n");
	printf("Reconstructing pinhole image\n");

	profileStart();
	profilePush("Tree gathering (pinhole)");

	vizDone |= (1<<viz);
	m_cameraParams.overrideUVT = Vec3f(0,0,1);

	switch(viz)
	{
	case VIZ_RECONSTRUCTION_PINHOLE:
		{
			m_cameraParams.reconstruction = RECONSTRUCTION_TRIANGLE2;
			Image* img = (m_cameraParams.enableCuda) ? m_reconstructionPinholeImageCuda : m_reconstructionPinholeImage;
			TreeGather filter(*m_samples, m_cameraParams, 1.f, m_focalDistance);
			filter.reconstructDofMotion(*img, m_debugImage);

			// scale debug data to [0,1]
			Vec4f mxVal(0);
			for(int y=0;y<m_debugImage->getSize().y;y++)
			for(int x=0;x<m_debugImage->getSize().x;x++)
			{
				mxVal = max(mxVal, m_debugImage->getVec4f(Vec2i(x,y)));
			}

			for(int y=0;y<m_debugImage->getSize().y;y++)
			for(int x=0;x<m_debugImage->getSize().x;x++)
			{
				Vec4f c = m_debugImage->getVec4f(Vec2i(x,y)) / mxVal;
				m_debugImage->setVec4f(Vec2i(x,y), c);
			}
			break;
		}

	default:
		fail("reconstructPinhole");
	}

	m_cameraParams.overrideUVT = Vec3f(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);

	profilePop();
	profileEnd();
}

//------------------------------------------------------------------------

void App::reconstruct(Visualization viz)
{
	U32& vizDone = (m_cameraParams.enableCuda ? m_vizDoneCuda : m_vizDone);
	if(!m_haveSampleBuffer || (vizDone&(1<<viz)))
		return;

	printf("\n");
	printf("Reconstructing 128spp image\n");

	vizDone |= (1<<viz);
	profileStart();
	profilePush("Tree gathering");

	switch(viz)
	{
	case VIZ_RECONSTRUCTION:
		{
			m_cameraParams.reconstruction = RECONSTRUCTION_TRIANGLE2;
			TreeGather filter(*m_samples, m_cameraParams, 1.f, m_focalDistance);
			Image* img = (m_cameraParams.enableCuda) ? m_reconstructionImageCuda : m_reconstructionImage;
			filter.reconstructDofMotion(*img);
			break;
		}
	}

	profilePop();
	profileEnd();
}

//------------------------------------------------------------------------

void App::render (GLContext* gl)
{
	if(!m_reconstructionImage)
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		return;
	}

	// Copy requested buffer to image.

	CameraParams params;
	Image img( *m_reconstructionImage );

	Visualization viz = (Visualization)m_showImage;

	switch(m_showImage)
	{
	case VIZ_INPUT:
		img = *m_inputImage;
		break;

	case VIZ_GROUNDTRUTH:
		img = *m_groundTruthImage;
		break;

	case VIZ_RECONSTRUCTION:
		if ( !m_displayNoCuda )
			reconstruct(viz);
		img = (m_cameraParams.enableCuda ? *m_reconstructionImageCuda : *m_reconstructionImage);
		break;

	case VIZ_GROUNDTRUTH_PINHOLE:
		img = *m_groundTruthPinholeImage;
		break;

	case VIZ_RECONSTRUCTION_PINHOLE:
		if ( !m_displayNoCuda )
			reconstructPinhole(viz);
		img = (m_cameraParams.enableCuda ? *m_reconstructionPinholeImageCuda : *m_reconstructionPinholeImage);
		break;

	case VIZ_NUM_SURFACES:
		reconstructPinhole(VIZ_RECONSTRUCTION_PINHOLE);
		for(int y=0;y<img.getSize().y;y++)
		for(int x=0;x<img.getSize().x;x++)
		{
			float c = m_debugImage->getVec4f(Vec2i(x,y))[0];
			img.setVec4f(Vec2i(x,y), Vec4f(c,c,c,1));
		}
		break;

	default:
		break;
	}	

	if ( m_showImage != VIZ_GROUNDTRUTH && m_showImage != VIZ_GROUNDTRUTH_PINHOLE )
		adjustGamma(img);

	blitToWindow(gl, img);

	if( m_displayNoCuda )
		gl->drawModalMessage( "Executable not built with FW_USE_CUDA (see README)" );
}

//------------------------------------------------------------------------

void App::blitToWindow(GLContext* gl, Image& img)
{
	m_window.setSize( img.getSize() );

	Mat4f oldXform = gl->setVGXform(Mat4f());
	glPushAttrib(GL_ENABLE_BIT);
	glDisable(GL_DEPTH_TEST);
	gl->drawImage(img, Vec2f(0.0f), 0.5f, m_flipY);
	gl->setVGXform(oldXform);
	glPopAttrib();
}

//------------------------------------------------------------------------

void App::adjustGamma(Image& img) const
{
	for(int y=0;y<img.getSize().y;y++)
	for(int x=0;x<img.getSize().x;x++)
	{
		Vec4f c = img.getVec4f(Vec2i(x,y));
		c.x = pow(c.x, 1/m_gamma);
		c.y = pow(c.y, 1/m_gamma);
		c.z = pow(c.z, 1/m_gamma);
		img.setVec4f(Vec2i(x,y), c);
	}
}

//------------------------------------------------------------------------

void App::exportAVI(const String& filename)
{
	// uv-movie
	Image image(m_reconstructionImage->getSize(), ImageFormat::RGBA_Vec4f);
    Image debug(m_reconstructionImage->getSize(), ImageFormat::RGBA_Vec4f);

	Vec2i videoSize = image.getSize()*Vec2i(2,1);
	videoSize.x = (videoSize.x+15) & -16;
	videoSize.y = (videoSize.y+15) & -16;
	Image frame(videoSize, ImageFormat::ABGR_8888);
	AviExporter avi(filename, videoSize, 5);

	frame.clear(0);

	const int numFrames = 4;		// per dimension
	for(int v=0;v<numFrames;v++)
	for(int u=0;u<numFrames;u++)
	{
		const int frameNum = (v*numFrames+u+1);
		printf("processing frame %d/%d\n", frameNum,numFrames*numFrames);

		// reconstruct

		if(v%2==0)	m_cameraParams.overrideUVT = Vec3f(float(u+0.5f)/numFrames,float(v+0.5f)/numFrames,0.5f);				// left to right
		else		m_cameraParams.overrideUVT = Vec3f(float(numFrames-1-u+0.5f)/numFrames,float(v+0.5f)/numFrames,0.5f);	// right to left
		TreeGather filter(*m_samples, m_cameraParams, 1.f, m_focalDistance);
		filter.reconstructDofMotion(image, &debug);

		if(!m_flipY)	// ehhh...
		{
			image.flipY();
			debug.flipY();
		}

		// scale debug data to [0,1]

		Vec4f mxVal(0);
		for(int y=0;y<debug.getSize().y;y++)
		for(int x=0;x<debug.getSize().x;x++)
		{
			mxVal = max(mxVal, debug.getVec4f(Vec2i(x,y)));
		}
		for(int y=0;y<debug.getSize().y;y++)
		for(int x=0;x<debug.getSize().x;x++)
		{
			Vec4f c = debug.getVec4f(Vec2i(x,y)) / mxVal;
			debug.setVec4f(Vec2i(x,y), c);
		}

		// left: reconstructed image

		for(int y=0;y<image.getSize().y;y++)
		for(int x=0;x<image.getSize().x;x++)
			frame.setABGR(Vec2i(x,y), image.getABGR(Vec2i(x,y)));

		// right: #surfaces
		for(int y=0;y<image.getSize().y;y++)
		for(int x=0;x<image.getSize().x;x++)
			frame.setVec4f(Vec2i(image.getSize().x+x,y), Vec4f(debug.getVec4f(Vec2i(x,y))[0]));

		// output to avi
		adjustGamma(frame);
		avi.getFrame() = frame;
		avi.exportFrame();

		// export to pngs as well

		String framename;
		framename = filename.getDirName() + "/Frame" + String(frameNum) + ".png";
		exportImage(framename.getPtr(), &frame);

		// show most recent frame in window (a bit awkward here).
//		adjustGamma(image);
//		blitToWindow(m_window.getGL(), image);
//	    m_window.repaint();
	}

	m_cameraParams.overrideUVT = Vec3f(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
}
