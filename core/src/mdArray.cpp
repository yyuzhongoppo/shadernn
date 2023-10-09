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
#include "snn/mdArray.h"
#include <algorithm>

namespace snn {

void preparePermute(uint32_t n, const std::vector<uint32_t>& dimIndicies, const uint32_t* dims, uint32_t* permutedDims,
                    const size_t* planeSizes, size_t* permutedPlaneSizes) {
    SNN_ASSERT(dimIndicies.size() <= n);
    std::vector<bool> remained(n, true);
    uint32_t i = 0;
    for (; i < dimIndicies.size(); ++i) {
        uint32_t ind = dimIndicies[i];
        SNN_ASSERT(ind < n);
        SNN_ASSERT(std::find(dimIndicies.begin() + i + 1, dimIndicies.end(), ind) == dimIndicies.end());
        permutedDims[i] = dims[ind];
        permutedPlaneSizes[i] = planeSizes[ind];
        remained[ind] = false;
    }
    for (uint32_t j = 0; j < n; ++j) {
        if (remained[j]) {
            SNN_ASSERT(i < n);
            permutedDims[i] = dims[j];
            permutedPlaneSizes[i] = planeSizes[j];
        }
    }
}

void computePlaneSizes(size_t n, const uint32_t* dims, size_t* planeSizes, size_t& size) {
    planeSizes[n - 1] = 1U;
    size = dims[n - 1];
    if (n > 1) {
        for (int i = n - 2; i >= 0; --i) {
            planeSizes[i] = planeSizes[i + 1] * dims[i + 1];
            size *= dims[i];
        }
    }
}

} // namespace snn