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
#include "gpu/GLContext.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Buffer;

//------------------------------------------------------------------------

class CudaModule
{
public:
                        CudaModule          (const void* cubin);
                        CudaModule          (const String& cubinFile);
                        ~CudaModule         (void);

    CUmodule            getHandle           (void) { return m_module; }

    Buffer&             getGlobal           (const String& name);
    void                updateGlobals       (bool async = false, CUstream stream = NULL); // copy to the device if modified

    CUfunction          getKernel           (const String& name, int paramSize = 0);
    int                 setParami           (CUfunction kernel, int offset, S32 value); // returns sizeof(value)
    int                 setParamf           (CUfunction kernel, int offset, F32 value);
    int                 setParamPtr         (CUfunction kernel, int offset, CUdeviceptr value);

    CUtexref            getTexRef           (const String& name);
    void                setTexRefMode       (CUtexref texRef, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                setTexRef           (const String& name, Buffer& buf, CUarray_format format, int numComponents);
    void                setTexRef           (const String& name, CUdeviceptr ptr, S64 size, CUarray_format format, int numComponents);
    void                setTexRef           (const String& name, CUarray cudaArray, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                setTexRef           (const String& name, const Image& image, bool wrap = true, bool bilinear = true, bool normalizedCoords = true, bool readAsInt = false);
    void                unsetTexRef         (const String& name);
    void                updateTexRefs       (CUfunction kernel);

    CUsurfref           getSurfRef          (const String& name);
    void                setSurfRef          (const String& name, CUarray cudaArray);

    void                launchKernel        (CUfunction kernel, const Vec2i& blockSize, const Vec2i& gridSize, bool async = false, CUstream stream = NULL);
    void                launchKernel        (CUfunction kernel, const Vec2i& blockSize, int numBlocks, bool async = false, CUstream stream = NULL) { launchKernel(kernel, blockSize, selectGridSize(numBlocks), async, stream); }
    F32                 launchKernelTimed   (CUfunction kernel, const Vec2i& blockSize, const Vec2i& gridSize, bool async = false, CUstream stream = NULL, bool yield = true);
    F32                 launchKernelTimed   (CUfunction kernel, const Vec2i& blockSize, int numBlocks, bool async = false, CUstream stream = NULL) { return launchKernelTimed(kernel, blockSize, selectGridSize(numBlocks), async, stream); }

    static void         staticInit          (void);
    static void         staticDeinit        (void);
    static bool         isAvailable         (void)      { staticInit(); return s_available; }
    static S64          getMemoryUsed       (void);
    static void         sync                (bool yield = true);
    static void         checkError          (const char* funcName, CUresult res);
    static const char*  decodeError         (CUresult res);

    static CUdevice     getDeviceHandle     (void)      { staticInit(); return s_device; }
    static int          getDriverVersion    (void); // e.g. 23 = 2.3
    static int          getComputeCapability(void); // e.g. 13 = 1.3
    static int          getDeviceAttribute  (CUdevice_attribute attrib);
    static bool         setPreferL1OverShared(bool preferL1) { bool old = s_preferL1; s_preferL1 = preferL1; return old; }

private:
    static CUdevice     selectDevice        (void);
    static void         printDeviceInfo     (CUdevice device);
    static Vec2i        selectGridSize      (int numBlocks);

private:
                        CudaModule          (const CudaModule&); // forbidden
    CudaModule&         operator=           (const CudaModule&); // forbidden

private:
    static bool         s_inited;
    static bool         s_available;
    static CUdevice     s_device;
    static CUcontext    s_context;
    static CUevent      s_startEvent;
    static CUevent      s_endEvent;
    static bool         s_preferL1;

    CUmodule            m_module;
    Array<Buffer*>      m_globals;
    Hash<String, S32>   m_globalHash;
    Array<CUtexref>     m_texRefs;
    Hash<String, S32>   m_texRefHash;
};

//------------------------------------------------------------------------
}
