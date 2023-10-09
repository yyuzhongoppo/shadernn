/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "snn/utils.h"
#include <array>
#include <vector>
#include <optional>

namespace snn {

template<typename T, uint32_t N, typename A>
class MdArray;

template<typename T, uint32_t N>
class MdSubArray;

template<typename T, uint32_t N>
class MdArrayView;

/// @brief Reduces the array of fixed size
/// @tparam A Data type
/// @tparam N Array size
/// @param arr Array 
/// @return Array's product
template<typename A, uint32_t N>
struct Reduce {
    static A product(const A* arr) {
        return arr[0] * Reduce<A, N - 1>::product(arr + 1);
    }
};

template<typename A>
struct Reduce<A, 1> {
    static A product(const A* arr) {
        return arr[0];
    }
};

typedef std::vector<std::pair<std::optional<uint32_t>, std::optional<uint32_t>>> MDArraySliceCoords;

/// @brief Helper functions to create Multidimensional array slice
/// @tparam T Data type
/// @tparam N Number of dimensions
/// @param ptr Data pointer
/// @param n Data size
/// @param origDims Original dimensions
/// @param origOffsets Original offsets
/// @param planeSizes Plane sizes
/// @param coords View coordinates
/// @return Multidimensional array slice
template<typename T, uint32_t N>
MdArrayView<T, N> createSlice(T* ptr, size_t n, const uint32_t* origDims, const size_t* origOffsets, const size_t* planeSizes,
                              const MDArraySliceCoords& coords) {
    SNN_ASSERT(coords.size() <= N);
    std::array<size_t, N> offsets;
    std::array<uint32_t, N> dims;
    size_t i = 0;
    T* ptr1 = ptr;
    size_t n1 = n;
    for (; i < coords.size(); ++i) {
        const auto& coord = coords[i];
        uint32_t start = coord.first.value_or(0);
        SNN_ASSERT(start <= origDims[i]);
        uint32_t end = coord.second.value_or(origDims[i]);
        SNN_ASSERT(end <= origDims[i]);
        SNN_ASSERT(start <= end);

        offsets[i] = (origOffsets[i] + start) * planeSizes[i];
        dims[i] = (end - start);
    }
    for (; i < N; ++i) {
        offsets[i] = origOffsets[i];
        dims[i] = origDims[i];
    }
    ptr1 += offsets[0];
    SNN_ASSERT(n >= offsets[0]);
    n1 -= offsets[0];
    return MdArrayView<T, N>(ptr1, n1, dims.data(), planeSizes, offsets);
}

template<typename T>
MdArrayView<T, 1> createSlice(T* ptr, size_t n, const MDArraySliceCoords& coords) {
    SNN_ASSERT(coords.size() <= 1);
    T* ptr1 = ptr;
    size_t n1 = n;
    if (!coords.empty()) {
        const auto& coord = coords[0];
        uint32_t start = coord.first.value_or(0);
        SNN_ASSERT(start <= n);
        uint32_t end = coord.second.value_or(n);
        SNN_ASSERT(end <= n);
        SNN_ASSERT(start <= end);
        ptr1 += start;
        n1 -= (size_t)(end - start);
    }
    return MdArrayView<T, 1>(ptr1, n1);
}

/// @brief Helper function to compute absolute offset from indicies and plane sizes
/// @tparam N Number of dimensions
/// @param ind Indicies
/// @param planeSizes Plane sizes
/// @return Absolute offset
template<uint32_t N>
size_t mdFlatIndicies(const uint32_t* ind, const size_t* planeSizes);

/// @brief Helper functions to prepare permute parameters
/// @param n Number of dimensions
/// @param dimIndicies Indicies of original dimensions in a permuted order
/// @param dims Original dimensions
/// @param permutedDims Permuted dimensions
/// @param planeSizes Original plane sizes
/// @param permutedPlaneSizes Permuted plane sizes
void preparePermute(uint32_t n, const std::vector<uint32_t>& dimIndicies, const uint32_t* dims, uint32_t* permutedDims,
                    const size_t* planeSizes, size_t* permutedPlaneSizes);

/// @brief Helper class. Currently does permute() and print() operations
/// @tparam T Data type
/// @tparam N Number of dimensions
/// @tparam T1 Converted data type
template<typename T, uint32_t N, typename T1 = T>
struct MdHelper {
    static void permute(const T* src, T1* target, const uint32_t* targetDims, const size_t* srcPlaneSizes, const size_t* targetPlaneSizes) {
        const uint32_t* targetDims1 = targetDims + 1;
        const size_t* srcPlaneSizes1 = srcPlaneSizes + 1;
        const size_t* targetPlaneSizes1 = targetPlaneSizes + 1;
        for (size_t i = 0; i < targetDims[0]; ++i) {
            MdHelper<T, N - 1, T1>::permute(src, target, targetDims1, srcPlaneSizes1, targetPlaneSizes1);
            target += targetPlaneSizes[0];
            src += srcPlaneSizes[0];
        }
    }

    static void print(const T* ptr, const uint32_t* dims, const size_t* planeSizes, FILE* fp = stdout) {
        const uint32_t* dims1 = dims + 1;
        const size_t* planeSizes1 = planeSizes + 1;
        for (size_t i = 0; i < dims[0]; ++i) {
            MdHelper<T, N - 1, T1>::print(ptr, dims1, planeSizes1, fp);
            ptr += planeSizes[0];
        }
        printf("\n");
    }
};

template<typename T, typename T1>
struct MdHelper<T, 1, T1> {
    static void permute(const T* src, T1* target, const uint32_t* targetDims, const size_t* srcPlaneSizes, const size_t*) {
        for (size_t i = 0; i < targetDims[0]; ++i) {
            *target = static_cast<T1>(*src);
            target++;
            src += srcPlaneSizes[0];
        }
    }

    static void print(const T* ptr, const uint32_t* dims, const size_t* , FILE* fp = stdout) {
        for (size_t i = 0; i < dims[0]; ++i) {
            printNumber(static_cast<T1>(ptr[i]), fp);
        }
        printf("\n");
    }
};

/// @brief Helper functions to permute dimensions of multidimensional array
/// @tparam T Data type
/// @tparam N Number of dimensions
/// @tparam T1 Converted data type
/// @tparam A Allocator
/// @param src Source buffer
/// @param dimIndicies Indicies of original dimensions in a permuted order
/// @param dims Original dimensions
/// @param planeSizes Original plane sizes
/// @return Multidimensional permuted array
template<typename T, uint32_t N, typename T1 = T, typename A = ArrayAllocator<T>>
MdArray<T1, N, A> mdPermute(T* src, const std::vector<uint32_t>& dimIndicies, const uint32_t* dims, const size_t* planeSizes) {
    SNN_ASSERT(dimIndicies.size() <= N);
    std::array<uint32_t, N> permutedDims;
    std::array<size_t, N> permutedPlaneSizes;
    preparePermute(N, dimIndicies, dims, permutedDims.data(), planeSizes, permutedPlaneSizes.data());
    MdArray<T1, N, A> permuted(permutedDims);
    MdHelper<T, N, T1>::permute(src, permuted.data(), permutedDims.data(), permutedPlaneSizes.data(), permuted.planeSizes().data());
    return permuted;
}

/// @brief Helper functions to permute dimensions of multidimensional array
/// @tparam T Data type
/// @tparam N Number of dimensions
/// @tparam T1 Converted data type
/// @tparam A Allocator
/// @param src Source buffer
/// @param target Target buffer
/// @param n Target buffer size
/// @param dimIndicies Indicies of original dimensions in a permuted order
/// @param dims Original dimensions
/// @param planeSizes Original plane sizes
/// @return Multidimensional permuted array
template<typename T, uint32_t N, typename T1 = T, typename A = ArrayAllocator<T1>>
MdArrayView<T1, N> mdPermute(T* src, T1* target, size_t n, const std::vector<uint32_t>& dimIndicies, const uint32_t* dims, const size_t* planeSizes) {
    SNN_ASSERT(dimIndicies.size() <= N);
    std::array<uint32_t, N> permutedDims;
    std::array<size_t, N> permutedPlaneSizes;
    preparePermute(N, dimIndicies, dims, permutedDims.data(), planeSizes, permutedPlaneSizes.data());
    MdArrayView<T1, N> permuted(target, n, permutedDims);
    MdHelper<T, N, T1>::permute(src, target, permutedDims.data(), permutedPlaneSizes.data(), permuted.planeSizes().data());
    return permuted;
}

/// @brief Helper function to compute plane sizes
/// @param n Number of dimensions
/// @param dims Dimensions
/// @param planeSizes Computed plane sizes
/// @param size Computed total size
/// @return 
void computePlaneSizes(size_t n, const uint32_t* dims, size_t* planeSizes, size_t& size);

/// @brief Multidimensional array that owns the data
/// @tparam T Data type
/// @tparam A Allocator type
/// @tparam N Number of dimensions
template<typename T, uint32_t N, typename A = ArrayAllocator<T>>
class MdArray : protected FixedSizeArray<T, A> {
    static_assert(N > 0);
private:
    std::array<uint32_t, N> _dims;
    std::array<size_t, N> _planeSizes;

public:
    MdArray(A a = A())
        : FixedSizeArray<T, A>(a)
    {}

    MdArray(const std::array<uint32_t, N>& dims, A a = A())
        : FixedSizeArray<T, A>(a)
        , _dims(dims)
    {
        size_t n;
        computePlaneSizes(N, _dims.data(), _planeSizes.data(), n);
        this->allocate(n);
    }

    SNN_NO_COPY(MdArray);

    // can move
    MdArray(MdArray&& that)
        : _dims(that._dims)
        , _planeSizes(that._planeSizes)
    {
        FixedSizeArray<T, A>::operator = (std::move(that));
    }

    MdArray& operator = (MdArray&& that) {
        if (this != &that)
        {
            FixedSizeArray<T, A>::operator = (std::move(that));
            _dims = that._dims;
            _planeSizes = that._planeSizes;
        }
        return *this;
    }

    auto size() const -> size_t { return this->_size; }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i < N);
        return _dims[i];
    }

    const std::array<uint32_t, N>& dims() const { return _dims; }
    const std::array<size_t, N>& planeSizes() const { return _planeSizes; }

    auto data() -> T* { return this->_ptr; }
    auto data() const -> const T* { return this->_ptr; }

    auto begin() -> T* { return this->_ptr; }
    auto begin() const -> const T* { return this->_ptr; }

    auto end() -> T* { return this->_ptr + this->_size; }
    auto end() const -> const T* { return this->_ptr + this->_size; }

    auto back() -> T& { return this->_ptr[this->_size - 1]; }
    auto back() const -> const T& { return this->_ptr[this->_size - 1]; }

    MdSubArray<T, N - 1> operator[] (size_t i) {
        SNN_ASSERT(i < _dims[0]);
        return MdSubArray<T, N - 1>(this->_ptr + _planeSizes[0] * i, _planeSizes[0], &_dims.data()[1], &_planeSizes.data()[1]);
    }

    MdSubArray<const T, N - 1> operator[] (size_t i) const {
        SNN_ASSERT(i < _dims[0]);
        return MdSubArray<const T, N - 1>(this->_ptr + _planeSizes[0] * i, _planeSizes[0], &_dims.data()[1], &_planeSizes.data()[1]);
    }

    auto operator[](const std::array<uint32_t, N>& ind) const -> const T& {
        size_t index = mdFlatIndicies<N>(ind.data(), _planeSizes.data());
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](const std::array<uint32_t, N>& ind) -> T& {
        size_t index = mdFlatIndicies<N>(ind.data(), _planeSizes.data());
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, N> view() {
        return MdArrayView<T, N>(this->_ptr, this->_size, _dims.data(), _planeSizes.data());
    }

    MdArrayView<const T, N> view() const {
        return MdArrayView<const T, N>(this->_ptr, this->_size, _dims.data(), _planeSizes.data());
    }

    template<uint32_t N1>
    MdArrayView<T, N1> reshape(const std::array<uint32_t, N1>& dims) {
        return MdArrayView<T, N1>(this->_ptr, this->_size, dims.data());
    }

    template<uint32_t N1>
    MdArrayView<const T, N1> reshape(const std::array<uint32_t, N1>& dims) const {
        return MdArrayView<const T, N1>(this->_ptr, this->_size, dims.data());
    }

    template<typename T1 = T>
    MdArray<T1, N> permute(const std::vector<uint32_t>& dimIndicies) const {
        return mdPermute<T, N, T1>(this->_ptr, dimIndicies, _dims.data(), _planeSizes.data());
    }

    template<typename T1 = T>
    MdArrayView<T1, N> permute(const std::vector<uint32_t>& dimIndicies, T1* ptr, size_t n) const {
        return mdPermute<T, N, T1>(this->_ptr, ptr, n, dimIndicies, _dims.data(), _planeSizes.data());
    }

    template<typename T1 = T>
    void print(FILE* fp = stdout) const {
        MdHelper<T, N, T1>::print(this->_ptr, _dims.data(), _planeSizes.data(), fp);
    }
};

template<typename T, typename A>
class MdArray<T, 1, A> : protected FixedSizeArray<T, A> {
public:
    MdArray(A a = A())
        : FixedSizeArray<T, A>(a)
    {}

    MdArray(const std::array<uint32_t, 1>& dims)
    {
        this->allocate(dims[0]);
    }

    MdArray(size_t n)
    {
        this->allocate(n);
    }

    MdArray(T* ptr, size_t n)
    {   
        this->_ptr = ptr;
        this->_size = n;
    }

    SNN_NO_COPY(MdArray);

    // can move
    MdArray(MdArray&& that)
    {
        FixedSizeArray<T, A>::operator = (std::move(that));
    }

    MdArray& operator = (MdArray&& that) {
        if (this != &that)
        {
            FixedSizeArray<T, A>::operator = (std::move(that));
        }
        return *this;
    }

    auto size() const -> size_t { return this->_size; }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i == 0);
        (void) i;
        return (uint32_t)this->_size;
    }

    auto data() -> T* { return this->_ptr; }
    auto data() const -> const T* { return this->_ptr; }

    auto begin() -> T* { return this->_ptr; }
    auto begin() const -> const T* { return this->_ptr; }

    auto end() -> T* { return this->_ptr + this->_size; }
    auto end() const -> const T* { return this->_ptr + this->_size; }

    auto back() -> T& { return this->_ptr[this->_size - 1]; }
    auto back() const -> const T& { return this->_ptr[this->_size - 1]; }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, 1> view() {
        return MdArrayView<T, 1>(this->_ptr, this->_size);
    }

    MdArrayView<const T, 1> view() const {
        return MdArrayView<const T, 1>(this->_ptr, this->_size);
    }

    template<typename T1 = T>
    void print(FILE* fp = stdout) const {
        MdHelper<T, 1, T1>::print(this->_ptr, (const uint32_t*)this->_size, nullptr, fp);
    }
};

/// @brief Lightweight object that references the plane of the existing MDArray
/// @tparam T Data type
/// @tparam N N Number of dimensions
template<typename T, uint32_t N>
class MdSubArray : protected FixedSizeArrayRef<T> {
    static_assert(N > 0);
protected:
    const uint32_t* _dims;
    const size_t* _planeSizes;

public:
    MdSubArray(T* ptr, size_t n, const uint32_t* dims, const size_t* planeSizes)
        : FixedSizeArrayRef<T>(ptr, n)
        , _dims(dims)
        , _planeSizes(planeSizes)
    {
        SNN_ASSERT(dims);
        SNN_ASSERT(planeSizes);
    }

    template<typename A>
    MdSubArray(const MdArray<T, N, A>& mArr)
        : FixedSizeArrayRef<T>(mArr.data(), mArr.size())
        , _dims(mArr.dims().data())
        , _planeSizes(mArr.planeSizes().data())
    {}

    template<typename A>
    MdSubArray(MdArray<T, N, A>& mArr)
        : FixedSizeArrayRef<T>(mArr.data(), mArr.size())
        , _dims(mArr.dims().data())
        , _planeSizes(mArr.planeSizes().data())
    {}

    auto size() const -> size_t { return this->_size; }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i < N);
        return _dims[i];
    }

    const uint32_t* dims() const { return _dims; }
    const size_t* planeSizes() const { return _planeSizes; }

    auto data() -> T* { return this->_ptr; }
    auto data() const -> const T* { return this->_ptr; }

    auto begin() -> T* { return this->_ptr; }
    auto begin() const -> const T* { return this->_ptr; }

    auto end() -> T* { return this->_ptr + this->_size; }
    auto end() const -> const T* { return this->_ptr + this->_size; }

    auto back() -> T& { return this->_ptr[this->_size - 1]; }
    auto back() const -> const T& { return this->_ptr[this->_size - 1]; }

    MdSubArray<T, N - 1> operator[](size_t i) {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(this->_size >= _planeSizes[0] * i);
        return MdSubArray<T, N - 1>(this->_ptr + _planeSizes[0] * i, _planeSizes[0], &_dims[1], &_planeSizes[1]);
    }

    MdSubArray<const T, N - 1> operator[] (size_t i) const {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(this->_size >= _planeSizes[0] * i);
        return MdSubArray<const T, N - 1>(this->_ptr + _planeSizes[0] * i, _planeSizes[0], &_dims[1], &_planeSizes[1]);
    }

    auto operator[](const std::array<uint32_t, N>& ind) const -> const T& {
        size_t index = mdFlatIndicies<N>(ind.data(), _planeSizes);
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](const std::array<uint32_t, N>& ind) -> T& {
        size_t index = mdFlatIndicies<N>(ind.data(), _planeSizes);
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, N> view() {
        return MdArrayView<T, N>(this->_ptr, this->_size, _dims, _planeSizes);
    }

    MdArrayView<const T, N> view() const {
        return MdArrayView<const T, N>(this->_ptr, this->_size, _dims, _planeSizes);
    }

    template<uint32_t N1>
    MdArrayView<T, N1> reshape(const std::array<uint32_t, N1>& dims) {
        return MdArrayView<T, N1>(this->_ptr, this->_size, dims.data());
    }

    template<uint32_t N1>
    MdArrayView<const T, N1> reshape(const std::array<uint32_t, N1>& dims) const {
        return MdArrayView<const T, N1>(this->_ptr, this->_size, dims.data());
    }

    template<typename T1 = T>
    MdArray<T1, N> permute(const std::vector<uint32_t>& dimIndicies) const {
        return mdPermute<T, N, T1>(this->_ptr, dimIndicies, _dims, _planeSizes);
    }

    template<typename T1 = T>
    MdArrayView<T1, N> permute(const std::vector<uint32_t>& dimIndicies, T1* ptr, size_t n) const {
        return mdPermute<T, N, T1>(this->_ptr, ptr, n, dimIndicies, _dims, _planeSizes);
    }

    template<typename T1 = T>
    void print(FILE* fp = stdout) const {
        MdHelper<T, N, T1>::print(this->_ptr, _dims, _planeSizes, fp);
    }
};

template<typename T>
class MdSubArray<T, 1> : protected FixedSizeArrayRef<T> {
public:
    MdSubArray(T* ptr)
        : FixedSizeArrayRef<T>(ptr, 1)
    {}

    MdSubArray(T* ptr, size_t n, const uint32_t*, const size_t*)
        : FixedSizeArrayRef<T>(ptr, n)
    {}

    template<typename A>
    MdSubArray(const MdArray<T, 1, A>& mArr)
        : FixedSizeArrayRef<T>(mArr.data(), mArr.size())
    {}

    auto size() const -> size_t { return this->_size; }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i == 0);
        (void) i;
        return this->_size;
    }

    auto data() -> T* { return this->_ptr; }
    auto data() const -> const T* { return this->_ptr; }

    auto begin() -> T* { return this->_ptr; }
    auto begin() const -> const T* { return this->_ptr; }

    auto end() -> T* { return this->_ptr + this->_size; }
    auto end() const -> const T* { return this->_ptr + this->_size; }

    auto back() -> T& { return this->_ptr[this->_size - 1]; }
    auto back() const -> const T& { return this->_ptr[this->_size - 1]; }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, 1> view() {
        return MdArrayView<T, 1>(this->_ptr, this->_size);
    }

    MdArrayView<const T, 1> view() const {
        return MdArrayView<const T, 1>(this->_ptr, this->_size);
    }

    template<typename T1 = T>
    void print(FILE* fp = stdout) {
        MdHelper<T, 1, T1>::print(this->_ptr, (const uint32_t*)this->_size, nullptr, fp);
    }
};

template<typename T, uint32_t N>
class MdArraySubView;

/// @brief Multidimensional array view. Provides indirect transformed access to underlying MDArray.
/// @tparam T Data type
/// @tparam N Number of dimensions
template<typename T, uint32_t N>
class MdArrayView : protected FixedSizeArrayRef<T> {
    static_assert(N > 1);
protected:
    std::array<uint32_t, N> _dims;
    std::array<size_t, N> _offsets;
    std::array<size_t, N> _planeSizes;

public:
    MdArrayView()
    {}

    MdArrayView(T* ptr, size_t n, const uint32_t* dims, const size_t* planeSizes, const std::array<size_t, N>& offsets = {})
        : FixedSizeArrayRef<T>(ptr, n)
        , _offsets(offsets)
    {
        SNN_ASSERT(dims);
        SNN_ASSERT(planeSizes);
        std::copy(dims, dims + N, _dims.begin());
        std::copy(planeSizes, planeSizes + N, _planeSizes.begin());
    }

    MdArrayView(T* ptr, size_t n, const uint32_t* dims, const std::array<size_t, N>& offsets = {})
        : FixedSizeArrayRef<T>(ptr, n)
        , _offsets(offsets)
    {
        SNN_ASSERT(dims);
        std::copy(dims, dims + N, _dims.begin());
        size_t nc;
        computePlaneSizes(N, _dims.data(), _planeSizes.data(), nc);
        SNN_ASSERT(n == nc);
    }

    MdArrayView(T* ptr, size_t n, const std::array<uint32_t, N>& dims, const std::array<size_t, N>& offsets = {})
        : FixedSizeArrayRef<T>(ptr, n)
        , _dims(dims)
        , _offsets(offsets)
    {
        size_t nc;
        computePlaneSizes(N, _dims.data(), _planeSizes.data(), nc);
        SNN_ASSERT(n == nc);
    }

    SNN_NO_COPY(MdArrayView);

    // can move
    MdArrayView(MdArrayView&& that)
        : _dims(that._dims)
        , _offsets(that._offsets)
        , _planeSizes(that._planeSizes)
    {
        FixedSizeArrayRef<T>::operator = (std::move(that));
    }

    MdArrayView& operator = (MdArrayView&& that) {
        if (this != &that)
        {
            _dims = that._dims;
            _offsets = that._offsets;
            _planeSizes = that._planeSizes;
            FixedSizeArrayRef<T>::operator = (std::move(that));
        }
        return *this;
    }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i < N);
        return _dims[i];
    }

    const std::array<uint32_t, N>& dims() const { return _dims; }
    const std::array<uint32_t, N>& offsets() const { return _offsets; }
    const std::array<size_t, N>& planeSizes() const { return _planeSizes; }

    MdArraySubView<T, N - 1> operator[](size_t i) {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(_planeSizes[0] * i + _offsets[1] < this->_size);
        return MdArraySubView<T, N - 1>(this->_ptr + _planeSizes[0] * i + _offsets[1],
                                        _planeSizes[0] - _offsets[1],
                                        &_dims.data()[1], &_offsets.data()[1], &_planeSizes.data()[1]);
    }

    MdArraySubView<const T, N - 1> operator[](size_t i) const {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(_planeSizes[0] * i + _offsets[1] < this->_size);
        return MdArraySubView<T, N - 1>(this->_ptr + _planeSizes[0] * i + _offsets[1],
                                        _planeSizes[0] - _offsets[1],
                                        &_dims.data()[1], &_offsets.data()[1], &_planeSizes.data()[1]);
    }

    MdArrayView<T, N> slice(const MDArraySliceCoords& coords) {
        return createSlice<T, N>(this->_ptr, this->_size, _dims.data(), _offsets.data(), _planeSizes.data(), coords);
    }

    MdArrayView<const T, N> slice(const MDArraySliceCoords& coords) const {
        return createSlice<const T, N>(this->_ptr, this->_size, _dims.data(), _offsets.data(), _planeSizes.data(), coords);
    }

    // TODO: Change this to create a permuted view
    template<typename T1 = T>
    MdArray<T1, N> permute(const std::vector<uint32_t>& dimIndicies) const {
        return mdPermute<T, N, T1>(this->_ptr, dimIndicies, _dims.data(), _planeSizes.data());
    }

    template<typename T1 = T>
    MdArrayView<T1, N> permute(const std::vector<uint32_t>& dimIndicies, T1* ptr, size_t n) const {
        return mdPermute<T, N, T1>(this->_ptr, ptr, n, dimIndicies, _dims.data(), _planeSizes.data());
    }
};

template<typename T>
class MdArrayView<T, 1> : protected FixedSizeArrayRef<T> {
public:
    MdArrayView()
    {}

    MdArrayView(T* ptr, size_t n)
        : FixedSizeArrayRef<T>(ptr, n)
    {}

    SNN_NO_COPY(MdArrayView);

    // can move
    MdArrayView(MdArrayView&& that)
    {
        FixedSizeArrayRef<T>::operator = (std::move(that));
    }

    MdArrayView& operator = (MdArrayView&& that) {
        if (this != &that)
        {
            FixedSizeArrayRef<T>::operator = (std::move(that));
        }
        return *this;
    }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i == 0);
        (void) i;
        return (uint32_t)this->_size;
    }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, 1> slice(const MDArraySliceCoords& coords) {
        return createSlice<T, 1>(this->_ptr, this->_size, coords);
    }

    MdArrayView<const T, 1> slice(const MDArraySliceCoords& coords) const {
        return createSlice<const T, 1>(this->_ptr, this->_size, coords);
    }
};

/// @brief Lightweight object that references the plane of the existing MDArrayView
/// @tparam T Data type
/// @tparam N Number of dimensions
template<typename T, uint32_t N>
class MdArraySubView : protected FixedSizeArrayRef<T> {
    static_assert(N > 1);
protected:
    const uint32_t* _dims;
    const size_t* _offsets;
    const size_t* _planeSizes;

public:
    MdArraySubView(T* ptr, size_t n, const uint32_t* dims, const size_t* offsets, const size_t* planeSizes)
        : FixedSizeArrayRef<T>(ptr, n)
        , _dims(dims)
        , _offsets(offsets)
        , _planeSizes(planeSizes)
    {
        SNN_ASSERT(dims);
        SNN_ASSERT(offsets);
        SNN_ASSERT(planeSizes);
    }

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i < N);
        return _dims[i];
    }

    MdArraySubView<T, N - 1> operator[](size_t i) {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(_planeSizes[0] * i + _offsets[1] < this->_size);
        return MdArraySubView<T, N - 1>(this->_ptr + _planeSizes[0] * i + _offsets[1],
                                        _planeSizes[0] - _offsets[1],
                                        &_dims[1], &_offsets[1], &_planeSizes[1]);
    }

    MdArraySubView<const T, N - 1> operator[](size_t i) const {
        SNN_ASSERT(i < _dims[0]);
        SNN_ASSERT(_planeSizes[0] * i + _offsets[1] < this->_size);
        return MdArraySubView<const T, N - 1>(this->_ptr + _planeSizes[0] * i + _offsets[1],
                                              _planeSizes[0] - _offsets[1],
                                              &_dims[1], &_offsets[1], &_planeSizes[1]);
    }

    MdArrayView<T, N> slice(const MDArraySliceCoords& coords) {
        return createSlice<T, N>(this->_ptr, this->_size, _dims, _offsets, _planeSizes, coords);
    }

    MdArrayView<const T, N> slice(const MDArraySliceCoords& coords) const {
        return createSlice<const T, N>(this->_ptr, this->_size, _dims, _offsets, _planeSizes, coords);
    }

    // TODO: Change this to create a permuted view
    template<typename T1 = T>
    MdArray<T1, N> permute(const std::vector<uint32_t>& dimIndicies) const {
        return mdPermute<T, N, T1>(this->_ptr, dimIndicies, _dims, _planeSizes);
    }

    template<typename T1 = T>
    MdArrayView<T1, N> permute(const std::vector<uint32_t>& dimIndicies, T1* ptr, size_t n) const {
        return mdPermute<T, N, T1>(this->_ptr, ptr, n, dimIndicies, _dims, _planeSizes);
    }
};

template<typename T>
class MdArraySubView<T, 1> : protected FixedSizeArrayRef<T> {
public:
    MdArraySubView(T* ptr)
        : FixedSizeArrayRef<T>(ptr, 1)
    {}

    MdArraySubView(T* ptr, size_t n, const uint32_t*, const size_t*, const size_t*)
        : FixedSizeArrayRef<T>(ptr, n)
    {}

    uint32_t dim(size_t i) const {
        SNN_ASSERT(i == 0);
        (void) i;
        return this->_size;
    }

    auto operator[](size_t index) const -> const T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }
    auto operator[](size_t index) -> T& {
        SNN_ASSERT(index < this->_size);
        return this->_ptr[index];
    }

    MdArrayView<T, 1> slice(const MDArraySliceCoords& coords) {
        return createSlice<T, 1>(this->_ptr, this->_size, coords);
    }

    MdArrayView<const T, 1> slice(const MDArraySliceCoords& coords) const {
        return createSlice<const T, 1>(this->_ptr, this->_size, coords);
    }
};

} // namespace snn