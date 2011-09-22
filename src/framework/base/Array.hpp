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
#include "base/Math.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Growing array, similar to stl::vector.

template <class T> class Array
{
private:
    enum
    {
        MinBytes        = 256,                                          // Minimum number of bytes to allocate when the first element is being added.
    };

public:

    // Constructors.

    inline              Array       (void);                             // Create an empty array. Memory is allocated lazily.
    inline explicit     Array       (const T& item);                    // Create an array containing one element.
    inline              Array       (const T* ptr, int size);           // Copy contents from the given memory location. If NULL, the elements are left uninitialized.
    inline              Array       (const Array<T>& other);            // Copy constructor.
    inline              ~Array      (void);

    // Array-wide getters.

    inline int          getSize     (void) const;                       // Returns the number of elements contained in the array.
    inline int          getCapacity (void) const;                       // Returns the number of elements currently allocated. Can be larger than getSize().
    inline const T*     getPtr      (int idx = 0) const;                // Returns a pointer to the specified element.
    inline T*           getPtr      (int idx = 0);
    inline int          getStride   (void) const;                       // Returns the size of one element in bytes.
    inline int          getNumBytes (void) const;                       // Returns the size of the entire array in bytes.

    // Element access.

    inline const T&     get         (int idx) const;                    // Returns a reference to the specified element.
    inline T&           get         (int idx);
    inline T            set         (int idx, const T& item);           // Overwrites the specified element and returns the old value.
    inline const T&     getFirst    (void) const;                       // Returns a reference to the first element.
    inline T&           getFirst    (void);
    inline const T&     getLast     (void) const;                       // Returns a reference to the last element.
    inline T&           getLast     (void);
    inline void         getRange    (int start, int end, T* ptr) const; // Copies a range of elements (start..end-1) into the given memory location.
    inline Array<T>     getRange    (int start, int end) const;         // Copies a range of elements (start..end-1) into a newly allocated Array.
    inline void         setRange    (int start, int end, const T* ptr); // Overwrites a range of elements (start..end-1) from the given memory location.
    inline void         setRange    (int start, const Array<T>& other); // Overwrites a range of elements from the given Array.

    // Array-wide operations that may shrink the allocation.

    inline void         reset       (int size = 0);                     // Discards old contents, and resets size & capacity exactly to the given value.
    inline void         setCapacity (int numElements);                  // Resizes the allocation for exactly the given number of elements. Does not modify contents.
    inline void         compact     (void);                             // Shrinks the allocation to the match the current size. Does not modify contents.
    inline void         set         (const T* ptr, int size);           // Discards old contents, and re-initializes the array from the given memory location.
    inline void         set         (const Array<T>& other);            // Discards old contents, and re-initializes the array by cloning the given Array.

    // Array-wide operations that can only grow the allocation.

    inline void         clear       (void);                             // Sets the size to zero. Does not shrink the allocation.
    inline void         resize      (int size);                         // Sets the size to the given value. Allocates more space if necessary.
    inline void         reserve     (int numElements);                  // Grows the allocation to contain at least the given number of elements. Does not modify contents.

    // Element addition. Allocates more space if necessary.

    inline T&           add         (void);                             // Adds one element and returns a reference to it.
    inline T&           add         (const T& item);                    // Adds one element and initializes it to the given value.
    inline T*           add         (const T* ptr, int size);           // Appends a number of elements from the given memory location. If NULL, the elements are left uninitialized.
    inline T*           add         (const Array<T>& other);            // Appends elements from the given Array.
    inline T&           insert      (int idx);                          // Inserts a new element at the given index and returns a reference to it. Shifts the following elements up.
    inline T&           insert      (int idx, const T& item);           // Inserts a new element at the given index and initializes it to the given value. Shifts the following elements up.
    inline T*           insert      (int idx, const T* ptr, int size);  // Inserts a number of elements from the given memory location. If NULL, the elements are left uninitialized.
    inline T*           insert      (int idx, const Array<T>& other);   // Inserts elements from the given Array.

    // Element removal. Does not shrink the allocation.

    inline T            remove      (int idx);                          // Removes the given element and returns its value. Shifts the following elements down.
    inline void         remove      (int start, int end);               // Removes a range of elements (start..end-1). Shifts the following elements down.
    inline T&           removeLast  (void);                             // Removes the last element and returns a reference to its value.
    inline T            removeSwap  (int idx);                          // Removes the given element and returns its value. Swaps in the last element to fill the vacant slot.
    inline void         removeSwap  (int start, int end);               // Removes a range of elements (start..end-1). Swaps in the N last element to fill the vacant slots.
    inline T*           replace     (int start, int end, int size);     // remove(start, end), insert(start, NULL, size)
    inline T*           replace     (int start, int end, const T* ptr, int size); // remove(start, end), insert(start, ptr, size)
    inline T*           replace     (int start, int end, const Array<T>& other); // remove(start, end), insert(start, other)

    // Element search.

    inline int          indexOf     (const T& item, int fromIdx = 0) const; // Finds the first element that equals the given value, or -1 if not found.
    inline int          lastIndexOf (const T& item) const;              // Finds the last element that equals the given value, or -1 if not found.
    inline int          lastIndexOf (const T& item, int fromIdx) const;
    inline bool         contains    (const T& item) const;              // Checks whether the array contains an element that equals the given value.
    inline bool         removeItem  (const T& item);                    // Finds the first element that equals the given value and removes it.

    // Operators.

    inline const T&     operator[]  (int idx) const;
    inline T&           operator[]  (int idx);
    inline Array<T>&    operator=   (const Array<T>& other);
    inline bool         operator==  (const Array<T>& other) const;
    inline bool         operator!=  (const Array<T>& other) const;

    // Type-specific utilities.

    static inline void  copy        (T* dst, const T* src, int size);   // Analogous to memcpy().
    static inline void  copyOverlap (T* dst, const T* src, int size);   // Analogous to memmove().

    // Internals.

private:
    inline void         init        (void);
    void            realloc     (int size);
    void                reallocRound(int size);

private:
    T*              m_ptr;
    S32             m_size;
    S32             m_alloc;
};

//------------------------------------------------------------------------

template <class T> Array<T>::Array(void)
{
    init();
}

//------------------------------------------------------------------------

template <class T> Array<T>::Array(const T& item)
{
    init();
    add(item);
}

//------------------------------------------------------------------------

template <class T> Array<T>::Array(const T* ptr, int size)
{
    init();
    set(ptr, size);
}

//------------------------------------------------------------------------

template <class T> Array<T>::Array(const Array<T>& other)
{
    init();
    set(other);
}

//------------------------------------------------------------------------

template <class T> Array<T>::~Array(void)
{
    delete[] m_ptr;
}

//------------------------------------------------------------------------

template <class T> int Array<T>::getSize(void) const
{
    return m_size;
}

//------------------------------------------------------------------------

template <class T> int Array<T>::getCapacity(void) const
{
    return m_alloc;
}

//------------------------------------------------------------------------

template <class T> const T* Array<T>::getPtr(int idx) const
{
    FW_ASSERT(idx >= 0 && idx <= m_size);
    return m_ptr + idx;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::getPtr(int idx)
{
    FW_ASSERT(idx >= 0 && idx <= m_size);
    return m_ptr + idx;
}

//------------------------------------------------------------------------

template <class T> int Array<T>::getStride(void) const
{
    return sizeof(T);
}

//------------------------------------------------------------------------

template <class T> int Array<T>::getNumBytes(void) const
{
    return getSize() * getStride();
}

//------------------------------------------------------------------------

template <class T> const T& Array<T>::get(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < m_size);
    return m_ptr[idx];
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::get(int idx)
{
    FW_ASSERT(idx >= 0 && idx < m_size);
    return m_ptr[idx];
}

//------------------------------------------------------------------------

template <class T> T Array<T>::set(int idx, const T& item)
{
    T& slot = get(idx);
    T old = slot;
    slot = item;
    return old;
}

//------------------------------------------------------------------------

template <class T> const T& Array<T>::getFirst(void) const
{
    return get(0);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::getFirst(void)
{
    return get(0);
}

//------------------------------------------------------------------------

template <class T> const T& Array<T>::getLast(void) const
{
    return get(getSize() - 1);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::getLast(void)
{
    return get(getSize() - 1);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::getRange(int start, int end, T* ptr) const
{
    FW_ASSERT(end <= m_size);
    copy(ptr, getPtr(start), end - start);
}

//------------------------------------------------------------------------

template <class T> Array<T> Array<T>::getRange(int start, int end) const
{
    FW_ASSERT(end <= m_size);
    return Array<T>(getPtr(start), end - start);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::setRange(int start, int end, const T* ptr)
{
    FW_ASSERT(end <= m_size);
    copy(getPtr(start), ptr, end - start);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::setRange(int start, const Array<T>& other)
{
    setRange(start, start + other.getSize(), other.getPtr());
}

//------------------------------------------------------------------------

template <class T> void Array<T>::reset(int size)
{
    clear();
    setCapacity(size);
    m_size = size;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::setCapacity(int numElements)
{
    int c = max(numElements, m_size);
    if (m_alloc != c)
        realloc(c);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::compact(void)
{
    setCapacity(0);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::set(const T* ptr, int size)
{
    reset(size);
    if (ptr)
        copy(getPtr(), ptr, size);
}

//------------------------------------------------------------------------

template <class T> void Array<T>::set(const Array<T>& other)
{
    if (&other != this)
        set(other.getPtr(), other.getSize());
}

//------------------------------------------------------------------------

template <class T> void Array<T>::clear(void)
{
    m_size = 0;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::resize(int size)
{
    FW_ASSERT(size >= 0);
    if (size > m_alloc)
        reallocRound(size);
    m_size = size;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::reserve(int numElements)
{
    if (numElements > m_alloc)
        realloc(numElements);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::add(void)
{
    return *add(NULL, 1);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::add(const T& item)
{
    T* slot = add(NULL, 1);
    *slot = item;
    return *slot;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::add(const T* ptr, int size)
{
    int oldSize = getSize();
    resize(oldSize + size);
    T* slot = getPtr(oldSize);
    if (ptr)
        copy(slot, ptr, size);
    return slot;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::add(const Array<T>& other)
{
    return replace(getSize(), getSize(), other);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::insert(int idx)
{
    return *replace(idx, idx, 1);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::insert(int idx, const T& item)
{
    T* slot = replace(idx, idx, 1);
    *slot = item;
    return *slot;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::insert(int idx, const T* ptr, int size)
{
    return replace(idx, idx, ptr, size);
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::insert(int idx, const Array<T>& other)
{
    return replace(idx, idx, other);
}

//------------------------------------------------------------------------

template <class T> T Array<T>::remove(int idx)
{
    T old = get(idx);
    replace(idx, idx + 1, 0);
    return old;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::remove(int start, int end)
    {
    replace(start, end, 0);
    }

//------------------------------------------------------------------------

template <class T> T& Array<T>::removeLast(void)
{
    FW_ASSERT(m_size > 0);
    m_size--;
    return m_ptr[m_size];
}

//------------------------------------------------------------------------

template <class T> T Array<T>::removeSwap(int idx)
{
    FW_ASSERT(idx >= 0 && idx < m_size);

    T old = get(idx);
    m_size--;
    if (idx < m_size)
        m_ptr[idx] = m_ptr[m_size];
    return old;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::removeSwap(int start, int end)
{
    FW_ASSERT(start >= 0);
    FW_ASSERT(start <= end);
    FW_ASSERT(end <= m_size);

    int oldSize = m_size;
    m_size += start - end;

    int copyStart = max(m_size, end);
    copy(m_ptr + start, m_ptr + copyStart, oldSize - copyStart);
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::replace(int start, int end, int size)
{
    FW_ASSERT(start >= 0);
    FW_ASSERT(start <= end);
    FW_ASSERT(end <= m_size);
    FW_ASSERT(size >= 0);

    int tailSize = m_size - end;
    int newEnd = start + size;
    resize(m_size + newEnd - end);

    copyOverlap(m_ptr + newEnd, m_ptr + end, tailSize);
    return m_ptr + start;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::replace(int start, int end, const T* ptr, int size)
{
    T* slot = replace(start, end, size);
    if (ptr)
        copy(slot, ptr, size);
    return slot;
}

//------------------------------------------------------------------------

template <class T> T* Array<T>::replace(int start, int end, const Array<T>& other)
{
    Array<T> tmp;
    const T* ptr = other.getPtr();
    if (&other == this)
    {
        tmp = other;
        ptr = tmp.getPtr();
    }
    return replace(start, end, ptr, other.getSize());
}

//------------------------------------------------------------------------

template <class T> int Array<T>::indexOf(const T& item, int fromIdx) const
{
    for (int i = max(fromIdx, 0); i < getSize(); i++)
        if (get(i) == item)
            return i;
    return -1;
}

//------------------------------------------------------------------------

template <class T> int Array<T>::lastIndexOf(const T& item) const
{
    return lastIndexOf(item, getSize() - 1);
}

//------------------------------------------------------------------------

template <class T> int Array<T>::lastIndexOf(const T& item, int fromIdx) const
{
    for (int i = min(fromIdx, getSize() - 1); i >= 0; i--)
        if (get(i) == item)
            return i;
    return -1;
}

//------------------------------------------------------------------------

template <class T> bool Array<T>::contains(const T& item) const
{
    return (indexOf(item) != -1);
}

//------------------------------------------------------------------------

template <class T> bool Array<T>::removeItem(const T& item)
{
    int idx = indexOf(item);
    if (idx == -1)
        return false;
    remove(idx);
    return true;
}

//------------------------------------------------------------------------

template <class T> const T& Array<T>::operator[](int idx) const
{
    return get(idx);
}

//------------------------------------------------------------------------

template <class T> T& Array<T>::operator[](int idx)
{
    return get(idx);
}

//------------------------------------------------------------------------

template <class T> Array<T>& Array<T>::operator=(const Array<T>& other)
{
    set(other);
    return *this;
}

//------------------------------------------------------------------------

template <class T> bool Array<T>::operator==(const Array<T>& other) const
{
    if (getSize() != other.getSize())
        return false;

    for (int i = 0; i < getSize(); i++)
        if (get(i) != other[i])
            return false;
    return true;
}

//------------------------------------------------------------------------

template <class T> bool Array<T>::operator!=(const Array<T>& other) const
{
    return (!operator==(other));
}

//------------------------------------------------------------------------

template <class T> void Array<T>::copy(T* dst, const T* src, int size)
{
    FW_ASSERT(size >= 0);
    if (!size)
        return;

    FW_ASSERT(dst && src);
    for (int i = 0; i < size; i++)
        dst[i] = src[i];
}

//------------------------------------------------------------------------

template <class T> void Array<T>::copyOverlap(T* dst, const T* src, int size)
{
    FW_ASSERT(size >= 0);
    if (!size)
        return;

    FW_ASSERT(dst && src);
    if (dst < src || dst >= src + size)
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
    else
        for (int i = size - 1; i >= 0; i--)
            dst[i] = src[i];
}

//------------------------------------------------------------------------

template <class T> void Array<T>::init(void)
{
    m_ptr = NULL;
    m_size = 0;
    m_alloc = 0;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::realloc(int size)
{
    FW_ASSERT(size >= 0);

    T* newPtr = NULL;
    if (size)
    {
        newPtr = new T[size];
        copy(newPtr, m_ptr, min(size, m_size));
    }

    delete[] m_ptr;
    m_ptr = newPtr;
    m_alloc = size;
}

//------------------------------------------------------------------------

template <class T> void Array<T>::reallocRound(int size)
{
    FW_ASSERT(size >= 0);
    int rounded = max((int)(MinBytes / sizeof(T)), 1);
    while (size > rounded)
        rounded <<= 1;
    realloc(rounded);
}

//------------------------------------------------------------------------

inline void Array<S8>::copy(S8* dst, const S8* src, int size)           { memcpy(dst, src, size * sizeof(S8)); }
inline void Array<U8>::copy(U8* dst, const U8* src, int size)           { memcpy(dst, src, size * sizeof(U8)); }
inline void Array<S16>::copy(S16* dst, const S16* src, int size)        { memcpy(dst, src, size * sizeof(S16)); }
inline void Array<U16>::copy(U16* dst, const U16* src, int size)        { memcpy(dst, src, size * sizeof(U16)); }
inline void Array<S32>::copy(S32* dst, const S32* src, int size)        { memcpy(dst, src, size * sizeof(S32)); }
inline void Array<U32>::copy(U32* dst, const U32* src, int size)        { memcpy(dst, src, size * sizeof(U32)); }
inline void Array<F32>::copy(F32* dst, const F32* src, int size)        { memcpy(dst, src, size * sizeof(F32)); }
inline void Array<S64>::copy(S64* dst, const S64* src, int size)        { memcpy(dst, src, size * sizeof(S64)); }
inline void Array<U64>::copy(U64* dst, const U64* src, int size)        { memcpy(dst, src, size * sizeof(U64)); }
inline void Array<F64>::copy(F64* dst, const F64* src, int size)        { memcpy(dst, src, size * sizeof(F64)); }

inline void Array<Vec2i>::copy(Vec2i* dst, const Vec2i* src, int size)  { memcpy(dst, src, size * sizeof(Vec2i)); }
inline void Array<Vec2f>::copy(Vec2f* dst, const Vec2f* src, int size)  { memcpy(dst, src, size * sizeof(Vec2f)); }
inline void Array<Vec3i>::copy(Vec3i* dst, const Vec3i* src, int size)  { memcpy(dst, src, size * sizeof(Vec3i)); }
inline void Array<Vec3f>::copy(Vec3f* dst, const Vec3f* src, int size)  { memcpy(dst, src, size * sizeof(Vec3f)); }
inline void Array<Vec4i>::copy(Vec4i* dst, const Vec4i* src, int size)  { memcpy(dst, src, size * sizeof(Vec4i)); }
inline void Array<Vec4f>::copy(Vec4f* dst, const Vec4f* src, int size)  { memcpy(dst, src, size * sizeof(Vec4f)); }

inline void Array<Mat2f>::copy(Mat2f* dst, const Mat2f* src, int size)  { memcpy(dst, src, size * sizeof(Mat2f)); }
inline void Array<Mat3f>::copy(Mat3f* dst, const Mat3f* src, int size)  { memcpy(dst, src, size * sizeof(Mat3f)); }
inline void Array<Mat4f>::copy(Mat4f* dst, const Mat4f* src, int size)  { memcpy(dst, src, size * sizeof(Mat4f)); }

//------------------------------------------------------------------------
}
