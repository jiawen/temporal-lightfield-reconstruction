/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
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
#include "base/Defs.hpp"

//------------------------------------------------------------------------

#if !defined FW_USE_CUDA
#define FW_USE_CUDA 0		// change this to 1 only if you have CUDA Toolkit 4.0 installed
#endif

#define FW_USE_GLEW 0

//------------------------------------------------------------------------

#if (FW_USE_CUDA)
#include <cuda.h>
#   pragma warning(push,3)
#       include <vector_functions.h> // float4, etc.
#   pragma warning(pop)
#endif

#if (!FW_CUDA)
#   define _WIN32_WINNT 0x0501
#   define WIN32_LEAN_AND_MEAN
#   define _WINMM_
#   include <windows.h>
#   undef min
#   undef max

#   pragma warning(push,3)
#   include <mmsystem.h>
#   pragma warning(pop)

#   define _SHLWAPI_
#   include <shlwapi.h>
#endif

//------------------------------------------------------------------------

namespace FW
{
#if (!FW_CUDA)
void    setCudaDLLName      (const String& name);
void    initDLLImports      (void);
void    initGLImports       (void);
void    deinitDLLImports    (void);
#endif
}

//------------------------------------------------------------------------
// CUDA definitions.
//------------------------------------------------------------------------

#if (!FW_USE_CUDA)
#   define CUDA_VERSION 2010
#   define CUDAAPI __stdcall

typedef enum { CUDA_SUCCESS = 0}        CUresult;
typedef struct { FW::S32 x, y; }        int2;
typedef struct { FW::S32 x, y, z; }     int3;
typedef struct { FW::S32 x, y, z, w; }  int4;
typedef struct { FW::F32 x, y; }        float2;
typedef struct { FW::F32 x, y, z; }     float3;
typedef struct { FW::F32 x, y, z, w; }  float4;
typedef struct { FW::F64 x, y; }        double2;
typedef struct { FW::F64 x, y, z; }     double3;
typedef struct { FW::F64 x, y, z, w; }  double4;

typedef void*   CUfunction;
typedef void*   CUmodule;
typedef int     CUdevice;
typedef size_t  CUdeviceptr;
typedef void*   CUcontext;
typedef void*   CUdevprop;
typedef int     CUdevice_attribute;
typedef int     CUjit_option;
typedef void*   CUtexref;
typedef void*   CUarray;
typedef int     CUarray_format;
typedef int     CUaddress_mode;
typedef int     CUfilter_mode;
typedef void*   CUstream;
typedef void*   CUevent;
typedef void*   CUDA_MEMCPY2D;
typedef void*   CUDA_MEMCPY3D;
typedef void*   CUDA_ARRAY_DESCRIPTOR;
typedef void*   CUDA_ARRAY3D_DESCRIPTOR;

#endif

#if (CUDA_VERSION < 3010)
typedef void* CUsurfref;
#endif

#if (CUDA_VERSION < 3020)
typedef unsigned int    CUsize_t;
#else
typedef size_t          CUsize_t;
#endif

//------------------------------------------------------------------------
// GL definitions.
//------------------------------------------------------------------------

#if (!FW_CUDA && FW_USE_GLEW)
#   define GL_FUNC_AVAILABLE(NAME) (NAME != NULL)
#   define GLEW_STATIC
#   include "3rdparty/glew/include/GL/glew.h"
#   include "3rdparty/glew/include/GL/wglew.h"
#   if FW_USE_CUDA
#   include <cudaGL.h>
#   endif

#elif (!FW_CUDA && !FW_USE_GLEW)
#   define GL_FUNC_AVAILABLE(NAME) (isAvailable_ ## NAME())
#   include <GL/gl.h>
#   if FW_USE_CUDA
#   include <cudaGL.h>
#   endif

typedef char            GLchar;
typedef ptrdiff_t       GLintptr;
typedef ptrdiff_t       GLsizeiptr;
typedef unsigned int    GLhandleARB;

#define GL_ALPHA32F_ARB                     0x8816
#define GL_ARRAY_BUFFER                     0x8892
#define GL_BUFFER_SIZE                      0x8764
#define GL_COLOR_ATTACHMENT0                0x8CE0
#define GL_COLOR_ATTACHMENT1                0x8CE1
#define GL_COLOR_ATTACHMENT2                0x8CE2
#define GL_COMPILE_STATUS                   0x8B81
#define GL_DEPTH_ATTACHMENT                 0x8D00
#define GL_ELEMENT_ARRAY_BUFFER             0x8893
#define GL_FRAGMENT_SHADER                  0x8B30
#define GL_FRAMEBUFFER                      0x8D40
#define GL_FUNC_ADD                         0x8006
#define GL_GENERATE_MIPMAP                  0x8191
#define GL_GEOMETRY_INPUT_TYPE_ARB          0x8DDB
#define GL_GEOMETRY_OUTPUT_TYPE_ARB         0x8DDC
#define GL_GEOMETRY_SHADER_ARB              0x8DD9
#define GL_GEOMETRY_VERTICES_OUT_ARB        0x8DDA
#define GL_INFO_LOG_LENGTH                  0x8B84
#define GL_INVALID_FRAMEBUFFER_OPERATION    0x0506
#define GL_LINK_STATUS                      0x8B82
#define GL_PIXEL_PACK_BUFFER                0x88EB
#define GL_PIXEL_UNPACK_BUFFER              0x88EC
#define GL_RENDERBUFFER                     0x8D41
#define GL_RGB32F                           0x8815
#define GL_RGBA32F                          0x8814
#define GL_RGBA32UI                         0x8D70
#define GL_RGBA_INTEGER                     0x8D99
#define GL_STATIC_DRAW                      0x88E4
#define GL_DYNAMIC_COPY                     0x88EA
#define GL_TEXTURE0                         0x84C0
#define GL_TEXTURE1                         0x84C1
#define GL_TEXTURE2                         0x84C2
#define GL_TEXTURE_3D                       0x806F
#define GL_TEXTURE_CUBE_MAP                 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X      0x8515
#define GL_UNSIGNED_SHORT_5_5_5_1           0x8034
#define GL_UNSIGNED_SHORT_5_6_5             0x8363
#define GL_VERTEX_SHADER                    0x8B31
#define GL_ARRAY_BUFFER_BINDING             0x8894
#define GL_READ_FRAMEBUFFER                 0x8CA8
#define GL_DRAW_FRAMEBUFFER                 0x8CA9
#define GL_TEXTURE_MAX_ANISOTROPY_EXT       0x84FE
#define GL_LUMINANCE32UI_EXT                0x8D74
#define GL_LUMINANCE_INTEGER_EXT            0x8D9C
#define GL_DEPTH_STENCIL_EXT                0x84F9
#define GL_RGBA16F                          0x881A
#define GL_R32F                             0x822E
#define GL_RG                               0x8227
#define GL_R16F                             0x822D
#define GL_RG16F                            0x822F
#define GL_RGBA32UI_EXT                     0x8D70
#define GL_RGBA_INTEGER_EXT                 0x8D99
#define GL_R16UI                            0x8234
#define GL_RG_INTEGER                       0x8228
#define GL_DEPTH_COMPONENT32                0x81A7
#define GL_DEPTH_COMPONENT32F               0x8CAC
#define GL_DEPTH_COMPONENT16                0x81A5
#define GL_DEPTH_COMPONENT24                0x81A6
#define GL_DEPTH24_STENCIL8_EXT             0x88F0
#define GL_DEPTH_STENCIL_EXT                0x84F9
#define GL_LUMINANCE32F_ARB                 0x8818
#define GL_TEXTURE_RENDERBUFFER_NV          0x8E55
#define GL_RENDERBUFFER_EXT                 0x8D41
#define GL_RENDERBUFFER_COVERAGE_SAMPLES_NV 0x8CAB
#define GL_RENDERBUFFER_COLOR_SAMPLES_NV    0x8E10

#define WGL_ACCELERATION_ARB                0x2003
#define WGL_ACCUM_BITS_ARB                  0x201D
#define WGL_ALPHA_BITS_ARB                  0x201B
#define WGL_AUX_BUFFERS_ARB                 0x2024
#define WGL_BLUE_BITS_ARB                   0x2019
#define WGL_DEPTH_BITS_ARB                  0x2022
#define WGL_DOUBLE_BUFFER_ARB               0x2011
#define WGL_DRAW_TO_WINDOW_ARB              0x2001
#define WGL_FULL_ACCELERATION_ARB           0x2027
#define WGL_GREEN_BITS_ARB                  0x2017
#define WGL_PIXEL_TYPE_ARB                  0x2013
#define WGL_RED_BITS_ARB                    0x2015
#define WGL_SAMPLES_ARB                     0x2042
#define WGL_STENCIL_BITS_ARB                0x2023
#define WGL_STEREO_ARB                      0x2012
#define WGL_SUPPORT_OPENGL_ARB              0x2010
#define WGL_TYPE_RGBA_ARB                   0x202B
#define WGL_NUMBER_OVERLAYS_ARB             0x2008
#define WGL_NUMBER_UNDERLAYS_ARB            0x2009

#endif

//------------------------------------------------------------------------

#if (!FW_CUDA)
#   define FW_DLL_IMPORT_RETV(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#   define FW_DLL_IMPORT_VOID(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#   define FW_DLL_DECLARE_RETV(RET, CALL, NAME, PARAMS, PASS)   bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   define FW_DLL_DECLARE_VOID(RET, CALL, NAME, PARAMS, PASS)   bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   if (FW_USE_CUDA)
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void);
#   else
#       define FW_DLL_IMPORT_CUDA(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#       define FW_DLL_IMPORT_CUV2(RET, CALL, NAME, PARAMS, PASS)    bool isAvailable_ ## NAME(void); RET CALL NAME PARAMS;
#   endif
#   include "base/DLLImports.inl"
#   undef FW_DLL_IMPORT_RETV
#   undef FW_DLL_IMPORT_VOID
#   undef FW_DLL_DECLARE_RETV
#   undef FW_DLL_DECLARE_VOID
#   undef FW_DLL_IMPORT_CUDA
#   undef FW_DLL_IMPORT_CUV2
#endif

//------------------------------------------------------------------------
