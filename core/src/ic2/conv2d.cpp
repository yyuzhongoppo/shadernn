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
#include "conv2d.h"
#include "layerFactory.h"

using namespace snn;
using namespace snn::dp;

void Conv2DDesc::parse(ModelParser& parser, int layerId) {
    GenericConvDesc::parse(parser, layerId);
    parser.getConvolutionLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, activation, (int&) kernelSize, (int&) stride, biases, weightsConv(),
                               useBatchNormalization, batchNormalization, leakyReluAlpha, paddingT, paddingB, paddingL, paddingR, paddingMode,
                               useMultiInputs);
    SNN_LOGD("useBatchNormalization: %d, useMultiInputs: %d, isRange01: %d, numOutputPlanes: %d, numInputPlanes: %d, activation: %s, leakyReluAlpha: %.2f,"
             " kernelSize: %d, stride: %d, %s,\n\t"
             "useUniformShaders: %d, padding: %s, %s, %s, %s, paddingMode: %s ",
             useBatchNormalization, useMultiInputs, isRange01, numOutputPlanes, numInputPlanes, activation.c_str(), leakyReluAlpha, kernelSize, stride,
             preferHp ? "FP16" : "FP32", useUniformShaders, paddingT.c_str(), paddingB.c_str(), paddingL.c_str(), paddingR.c_str(), paddingMode.c_str());

}

void Conv2DLayer::getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const {
    GenericModelLayer::getOutputDims(width, height, depth);
    depth = _desc.numOutputPlanes;
}

void Conv2DLayer::getPaddingOffset(uint32_t (&offsets)[4]) const {
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

bool Conv2DLayer::oihw2hwo4i4(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit) {
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh * ROUND_UP(inChannels, unit);
    PROFILE_TIME(oihw2hwo4i4, "oihw2hwo4i4")

    SNN_LOGD("inChannels = %d, outChannels = %d, fw = %d, fh = %d, all: %d", inChannels, outChannels, fw, fh, alignedWeightSize);

    SNN_ASSERT(outChannels <= inputWeights.dim(0));
    SNN_ASSERT(inChannels <= inputWeights.dim(1));
    SNN_ASSERT(fh <= inputWeights.dim(2));
    SNN_ASSERT(fw <= inputWeights.dim(3));

    outVec.resize(alignedWeightSize, 0.0f);
    const uint32_t planeSize = ROUND_UP(outChannels, unit) * ROUND_UP(inChannels, unit);
    const uint32_t inSize = ROUND_UP(inChannels, unit) * unit;
    for (uint32_t oc = 0; oc < outChannels; ++oc) {      // oc:   output channel
        uint32_t od   = oc / unit;                       // od:   output depth
        uint32_t od_c = oc % unit;                       // od_c: output depth RGBA channel
        const auto& iw0 = inputWeights[oc];
        (void) iw0;
        for (uint32_t ic = 0; ic < inChannels; ++ic) {   // ic:   input channel
            const auto& iw1 = iw0[ic];
            (void) iw1;
            for (uint32_t y = 0; y < fh; ++y) {
                uint32_t base = y * fw * planeSize;
                const auto& iw2 = iw1[y];
                (void) iw2;
                for (uint32_t x = 0; x < fw; ++x, base += planeSize) {
#if 0
                    outVec[base + inSize * od + ic * unit + od_c] = iw2[x];
#else
                    outVec[base + inSize * od + ic * unit + od_c] = inputWeights.data()[oc * inChannels * fh * fw + ic * fh * fw + y * fw + x];
#endif
                }
            }
        }
    }
    return 0;
}

bool Conv2DLayer::oihw2hwo4i4fp16(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit) {
    int alignedWeightSize = ROUND_UP(outChannels, unit) * fw * fh * ROUND_UP(inChannels, unit)/2;

    SNN_ASSERT(outChannels <= inputWeights.dim(0));
    SNN_ASSERT(inChannels <= inputWeights.dim(1));
    SNN_ASSERT(fh <= inputWeights.dim(2));
    SNN_ASSERT(fw <= inputWeights.dim(3));

    outVec.resize(alignedWeightSize);
    const uint32_t planeSize = ROUND_UP(outChannels, unit) * ROUND_UP(inChannels, unit);
    const uint32_t inSize = ROUND_UP(inChannels, unit) * unit;
    uint16_t* out    = (uint16_t*) outVec.data();
    memset(out, 0, alignedWeightSize * sizeof(float));
    for (uint32_t oc = 0; oc < outChannels; ++oc) {
        uint32_t od   = oc / unit;                       // od:   output depth
        uint32_t od_c = oc % unit;                       // od_c: output depth RGBA channel
        const auto& iw0 = inputWeights[oc];
        for (uint32_t ic = 0; ic < inChannels; ++ic) {
            const auto& iw1 = iw0[ic];
            for (uint32_t y = 0; y < fh; ++y) {
                uint32_t base = y * fw * planeSize;
                const auto& iw2 = iw1[y];
                for (uint32_t x = 0; x < fw; ++x, base += planeSize) {
                    out[base + inSize * od + ic * unit + od_c] = FP32::toHalf(iw2[x]);
                }
            }
        }
    }
    return 0;
}

InferenceGraph::Transform Conv2DLayer::getOutputScaleDimAdjustment() const {
    uint32_t offset[4];
    getPaddingOffset(offset);
    float scale       = 1 / static_cast<float>(_desc.stride);
    float translation = 0.0f;
    if (_desc.kernelSize % 2 != 0) {
        translation = 1 + (static_cast<float>(offset[0] + offset[1]) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    } else {
        translation = 1 + (static_cast<float>(offset[0] + offset[1] - 1) - static_cast<float>(_desc.kernelSize)) / static_cast<float>(_desc.stride);
    }
    return {0, {{scale, scale, translation, translation}} };
}


