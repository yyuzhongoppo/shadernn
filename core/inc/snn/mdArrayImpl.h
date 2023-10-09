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
#include "snn/mdArray.h"
#include "snn/mdArrayImpl.h"
#include "snn/utils.h"
#include "snn/image.h"
#include <array>

namespace snn {

template<uint32_t N>
size_t mdFlatIndicies(const uint32_t* ind, const size_t* planeSizes) {
    return ind[0] * planeSizes[0] + mdFlatIndicies<N - 1>(ind + 1, planeSizes + 1);
}

template<>
size_t mdFlatIndicies<1>(const uint32_t* ind, const size_t*) {
    return ind[0];
}

template<>
struct MdHelper<float, 1, snn::FP16> {
    static void permute(const float* src, snn::FP16* target, const uint32_t* targetDims, const size_t* srcPlaneSizes, const size_t*) {
        for (size_t i = 0; i < targetDims[0]; ++i) {
            *target = FP32::toHalf(*src);
            target++;
            src += srcPlaneSizes[0];
        }
    }

    static void print(const float* ptr, const uint32_t* dims, const size_t* , FILE* fp = stdout) {
        for (size_t i = 0; i < dims[0]; ++i) {
            printNumber(FP32::toHalf(ptr[i]), fp);
        }
        printf("\n");
    }
};

} // namespace snn