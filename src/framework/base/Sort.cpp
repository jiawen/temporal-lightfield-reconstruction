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

#include "base/Sort.hpp"
#include "base/MulticoreLauncher.hpp"

using namespace FW;

//------------------------------------------------------------------------

#define QSORT_STACK_SIZE    32
#define QSORT_MIN_SIZE      16
#define MULTICORE_MIN_SIZE  (1 << 14)

//------------------------------------------------------------------------

namespace FW
{

struct TaskSpec
{
    S32             low;
    S32             high;
    void*           data;
    SortCompareFunc compareFunc;
    SortSwapFunc    swapFunc;
};

static inline void  insertionSort   (int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);
static inline int   median3         (int low, int high, void* data, SortCompareFunc compareFunc);
static int          partition       (int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);
static void         qsort           (int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);
static void         qsortMulticore  (MulticoreLauncher::Task& task);

}

//------------------------------------------------------------------------

void FW::insertionSort(int start, int size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    FW_ASSERT(compareFunc && swapFunc);
    FW_ASSERT(size >= 0);

    for (int i = 1; i < size; i++)
    {
        int j = start + i - 1;
        while (j >= start && compareFunc(data, j, j + 1) > 0)
        {
            swapFunc(data, j, j + 1);
            j--;
        }
    }
}

//------------------------------------------------------------------------

int FW::median3(int low, int high, void* data, SortCompareFunc compareFunc)
{
    FW_ASSERT(compareFunc);
    FW_ASSERT(low >= 0 && high >= 2);

    int l = low;
    int c = (low + high) >> 1;
    int h = high - 2;

    if (compareFunc(data, l, h) > 0) swap(l, h);
    if (compareFunc(data, l, c) > 0) c = l;
    return (compareFunc(data, c, h) > 0) ? h : c;
}

//------------------------------------------------------------------------

int FW::partition(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    // Select pivot using median-3, and hide it in the highest entry.

    swapFunc(data, median3(low, high, data, compareFunc), high - 1);

    // Partition data.

    int i = low - 1;
    int j = high - 1;
    for (;;)
    {
        do
            i++;
        while (compareFunc(data, i, high - 1) < 0);
        do
            j--;
        while (compareFunc(data, j, high - 1) > 0);

        FW_ASSERT(i >= low && j >= low && i < high && j < high);
        if (i >= j)
            break;

        swapFunc(data, i, j);
    }

    // Restore pivot.

    swapFunc(data, i, high - 1);
    return i;
}

//------------------------------------------------------------------------

void FW::qsort(int low, int high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    FW_ASSERT(compareFunc && swapFunc);
    FW_ASSERT(low <= high);

    int stack[QSORT_STACK_SIZE];
    int sp = 0;
    stack[sp++] = high;

    while (sp)
    {
        high = stack[--sp];
        FW_ASSERT(low <= high);

        // Small enough or stack full => use insertion sort.

        if (high - low < QSORT_MIN_SIZE || sp + 2 > QSORT_STACK_SIZE)
        {
            insertionSort(low, high - low, data, compareFunc, swapFunc);
            low = high + 1;
            continue;
        }

        // Partition and sort sub-partitions.

        int i = partition(low, high, data, compareFunc, swapFunc);
        FW_ASSERT(sp + 2 <= QSORT_STACK_SIZE);
        if (high - i > 2)
            stack[sp++] = high;
        if (i - low > 1)
            stack[sp++] = i;
        else
            low = i + 1;
    }
}

//------------------------------------------------------------------------

void FW::qsortMulticore(MulticoreLauncher::Task& task)
{
    // Small enough => use sequential qsort.

    TaskSpec* spec = (TaskSpec*)task.data;
    if (spec->high - spec->low < MULTICORE_MIN_SIZE)
        qsort(spec->low, spec->high, spec->data, spec->compareFunc, spec->swapFunc);

    // Otherwise => partition and schedule sub-partitions.

    else
    {
        int i = partition(spec->low, spec->high, spec->data, spec->compareFunc, spec->swapFunc);
        if (i - spec->low >= 2)
        {
            TaskSpec* childSpec = new TaskSpec(*spec);
            childSpec->high = i;
            task.launcher->push(qsortMulticore, childSpec);
        }
        if (spec->high - i > 2)
        {
            TaskSpec* childSpec = new TaskSpec(*spec);
            childSpec->low = i + 1;
            task.launcher->push(qsortMulticore, childSpec);
        }
    }

    // Free task spec.

    delete spec;
}

//------------------------------------------------------------------------

void FW::sort(int start, int end, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    FW_ASSERT(start <= end);
    FW_ASSERT(compareFunc && swapFunc);
    if (end - start < 2)
        return;

    qsort(start, end, data, compareFunc, swapFunc);
}

//------------------------------------------------------------------------

void FW::sortMulticore(int start, int end, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
    FW_ASSERT(start <= end);
    FW_ASSERT(compareFunc && swapFunc);
    if (end - start < 2)
        return;

    TaskSpec* spec = new TaskSpec;
    spec->low = start;
    spec->high = end;
    spec->data = data;
    spec->compareFunc = compareFunc;
    spec->swapFunc = swapFunc;

    MulticoreLauncher launcher;
    MulticoreLauncher::Task task;
    task.launcher = &launcher;
    task.data = spec;
    qsortMulticore(task);
}

//------------------------------------------------------------------------

int FW::compareU32(void* data, int idxA, int idxB)
{
    U32 a = ((U32*)data)[idxA];
    U32 b = ((U32*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapU32(void* data, int idxA, int idxB)
{
    swap(((U32*)data)[idxA], ((U32*)data)[idxB]);
}

//------------------------------------------------------------------------

int FW::compareU64(void* data, int idxA, int idxB)
{
    U64 a = ((U64*)data)[idxA];
    U64 b = ((U64*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapU64(void* data, int idxA, int idxB)
{
    swap(((U64*)data)[idxA], ((U64*)data)[idxB]);
}

//------------------------------------------------------------------------

int FW::compareS32(void* data, int idxA, int idxB)
{
    S32 a = ((S32*)data)[idxA];
    S32 b = ((S32*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapS32(void* data, int idxA, int idxB)
{
    swap(((S32*)data)[idxA], ((S32*)data)[idxB]);
}

//------------------------------------------------------------------------

int FW::compareS64(void* data, int idxA, int idxB)
{
    S64 a = ((S64*)data)[idxA];
    S64 b = ((S64*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapS64(void* data, int idxA, int idxB)
{
    swap(((S64*)data)[idxA], ((S64*)data)[idxB]);
}

//------------------------------------------------------------------------

int FW::compareF32(void* data, int idxA, int idxB)
{
    F32 a = ((F32*)data)[idxA];
    F32 b = ((F32*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapF32(void* data, int idxA, int idxB)
{
    swap(((F32*)data)[idxA], ((F32*)data)[idxB]);
}

//------------------------------------------------------------------------

int FW::compareF64(void* data, int idxA, int idxB)
{
    F64 a = ((F64*)data)[idxA];
    F64 b = ((F64*)data)[idxB];
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

void FW::swapF64(void* data, int idxA, int idxB)
{
    swap(((F64*)data)[idxA], ((F64*)data)[idxB]);
}

//------------------------------------------------------------------------
