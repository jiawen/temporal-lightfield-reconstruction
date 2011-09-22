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

#include "base/Timer.hpp"
#include "base/Thread.hpp"

using namespace FW;

//------------------------------------------------------------------------

F64 Timer::s_ticksToSecsCoef    = -1.0;
S64 Timer::s_prevTicks          = 0;

//------------------------------------------------------------------------

F32 Timer::end(void)
{
    S64 elapsed = getElapsedTicks();
    m_startTicks += elapsed;
    m_totalTicks += elapsed;
    return ticksToSecs(elapsed);
}

//------------------------------------------------------------------------

S64 Timer::queryTicks(void)
{
    if (!Thread::isMain())
        fail("Timers can only be used in the main thread!");

    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks))
        failWin32Error("QueryPerformanceCounter");

    s_prevTicks = max(s_prevTicks, ticks.QuadPart);
    return s_prevTicks;
}

//------------------------------------------------------------------------

F32 Timer::ticksToSecs(S64 ticks)
{
    if (s_ticksToSecsCoef == -1.0)
    {
        LARGE_INTEGER freq;
        if (!QueryPerformanceFrequency(&freq))
            failWin32Error("QueryPerformanceFrequency");
        s_ticksToSecsCoef = max(1.0 / (F64)freq.QuadPart, 0.0);
    }

    return (F32)(ticks * s_ticksToSecsCoef);
}

//------------------------------------------------------------------------

S64 Timer::getElapsedTicks(void)
{
    S64 curr = queryTicks();
    if (m_startTicks == -1)
        m_startTicks = curr;
    return curr - m_startTicks;
}

//------------------------------------------------------------------------
