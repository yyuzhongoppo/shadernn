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
#include "pch.h"
#include "separableconvolution.h"
#include "layerFactory.h"
#include "inferencepass.h"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

using namespace snn;
using namespace snn::dp;

void SeparableConv2DLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = _desc.paddingT;
    std::string paddingB = _desc.paddingB;
    std::string paddingL = _desc.paddingL;
    std::string paddingR = _desc.paddingR;
    bool isdigit         = std::all_of(paddingT.begin(), paddingT.end(), ::isdigit);
    if (isdigit) {
        offsets[0] = std::stoul(paddingT);
        offsets[1] = std::stoul(paddingB);
        offsets[2] = std::stoul(paddingL);
        offsets[3] = std::stoul(paddingR);
    } else {
        if (paddingT == "valid" || paddingT == "none") {
            offsets[0] = 0;
            offsets[1] = 0;
            offsets[2] = 0;
            offsets[3] = 0;
        } else {
            if (_desc.kernelSize > 1) {
                offsets[0] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[1] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[2] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                offsets[3] = std::max(static_cast<uint32_t>(_desc.kernelSize / 2), (uint32_t) 1);
                if (_desc.kernelSize % 2 == 0) {
                    offsets[0] = offsets[0] - 1;
                    offsets[2] = offsets[2] - 1;
                }
            } else {
                offsets[0] = 0;
                offsets[1] = 0;
                offsets[2] = 0;
                offsets[3] = 0;
            }
        }
    }
}

InferenceGraph::Transform SeparableConv2DLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    getPaddingOffset(offset);
    float scale       = 1.0f / static_cast<float>(_desc.stride);
    float translation = 0.0f;
    if (_desc.kernelSize % 2 != 0) {
        translation = 1.0f + (static_cast<float>(offset[0] + offset[1]) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    } else {
        translation = 1.0f + (static_cast<float>(offset[0] + offset[1] - 1) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    }
    return {0, {{scale, scale, translation, translation}} };
}

void SeparableConv2DLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);
    for (auto& dim : inputDims) {
        width  = (dim.width - _desc.kernelSize + paddingOffsets[0] + paddingOffsets[2]) / _desc.stride + 1;
        height = (dim.height - _desc.kernelSize + paddingOffsets[1] + paddingOffsets[3]) / _desc.stride + 1;
        depth  = dim.depth;
        break;
    }
}

bool SeparableConv2DLayer::oihw2hwo4i4(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels,
    int outChannels, int fw, int fh, int unit) {
    (void) inChannels;
    
    SNN_ASSERT(outChannels <= inputWeights.dim(1));
    SNN_ASSERT(fh <= inputWeights.dim(2));
    SNN_ASSERT(fw <= inputWeights.dim(3));

    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh;

    outVec.resize(alignedWeightSize, 0.0f);
    std::fill(outVec.begin(), outVec.end(), 0);
    const uint32_t planeSize = ROUND_UP(outChannels, unit) * fw;
    const uint32_t inSize = ROUND_UP(outChannels, unit);
    const auto& iw0 = inputWeights[0];
    for (uint32_t oc = 0; oc < outChannels; ++oc) {
        uint32_t od   = oc / unit;                       // od:   output depth
        uint32_t od_c = oc % unit;                       // od_c: output depth RGBA channel
        const auto& iw1 = iw0[oc];
        for (uint32_t y = 0; y < fh; ++y) {
            uint32_t base  = y * planeSize;
            const auto& iw2 = iw1[y];
            for (uint32_t x = 0; x < fw; ++x) {
                outVec[base + inSize * x + od * unit + od_c] = iw2[x];
            }
        }
    }
    return 0;
}
