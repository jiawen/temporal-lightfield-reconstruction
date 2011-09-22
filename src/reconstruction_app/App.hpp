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
#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "reconstruction/Reconstruction.hpp"

namespace FW
{
//------------------------------------------------------------------------

class App : public Window::Listener, public CommonControls::StateObject
{
private:
    enum Action
    {
        Action_None,
		Action_LoadSampleBuffer,
		Action_SaveSampleBuffer,
		Action_ExportUVSweep,
		Action_ClearImages,
		Action_RunSweep,
    };

	enum Visualization
	{
		VIZ_INPUT,
		VIZ_GROUNDTRUTH,
		VIZ_RECONSTRUCTION,	
		VIZ_GROUNDTRUTH_PINHOLE,
		VIZ_RECONSTRUCTION_PINHOLE,
		VIZ_NUM_SURFACES,
	};

public:
                    App             	(void);
    virtual         ~App            	(void);

    virtual bool    handleEvent     	(const Window::Event& ev);
    virtual void    readState       	(StateDump& d);
    virtual void    writeState      	(StateDump& d) const;

private:
    void            render				(GLContext* gl);
    void            importSampleBuffer	(const String& fileName);
	void			exportAVI			(const String& fileName);

	void			reconstructPinhole	(Visualization viz);
	void			reconstruct			(Visualization viz);

	void			blitToWindow		(GLContext* gl, Image& img);
	void			adjustGamma			(Image& image) const;
    void            firstTimeInit   	(void);

private:
                    App             	(const App&); // forbidden
    App&            operator=       	(const App&); // forbidden

private:
    Window          	m_window;
    CommonControls  	m_commonCtrl;
	CameraParams		m_cameraParams;

    Action          	m_action;

	UVTSampleBuffer*	m_samples;
	Image*				m_inputImage;
	Image*				m_groundTruthImage;
	Image*				m_reconstructionImage;
	Image*				m_reconstructionImageCuda;
	Image*				m_groundTruthPinholeImage;
	Image*				m_reconstructionPinholeImage;
	Image*				m_reconstructionPinholeImageCuda;
	Image*				m_debugImage;

	Image*				m_imageC;
	Image*				m_pinholeImageT;
	Image*				m_pinholeImageC;
	Image*				m_debugImageT;

	String				m_fileName;
	bool				m_haveSampleBuffer;
	U32					m_vizDone;
	U32					m_vizDoneCuda;

	bool				m_flipY;
	float				m_gamma;
	float				m_focalDistance;
	S32					m_showImage;

	bool				m_displayNoCuda;	// show a prompt if exe not built with CUDA and trying to enable it
};

//------------------------------------------------------------------------
}
