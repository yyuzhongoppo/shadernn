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

#include "genericlayer.h"
#include "snn/snn.h"
#include "modelparser.h"
#include "conv2dSupport.h"
#include <string>
#include <vector>
#include <map>
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

struct SeparableConv2DDesc : GenericConvDesc {
    bool useBatchNormalization;
    std::map<std::string, std::vector<float>> batchNormalization;
    float leakyReluAlpha;
    std::string padding;
    bool useUniformShaders    = true;
    std::string paddingT, paddingB, paddingL, paddingR;
    void parse(ModelParser& parser, int layerId) {
        GenericConvDesc::parse(parser, layerId);
        parser.getDepthwiseConvolutionLayer(layerId, (int&) numOutputPlanes, (int&) numInputPlanes, activation, (int&) kernelSize, (int&) stride, biases, weightsConv(),
        useBatchNormalization, batchNormalization, leakyReluAlpha, paddingT, paddingB, paddingL, paddingR);
        SNN_ASSERT(numOutputPlanes = numInputPlanes);
    }
};

// This is a base class to generates a shader for separable convolution
class SeparableConv2DLayer : public GenericConvolutionLayer {
public:
    SeparableConv2DLayer(SeparableConv2DDesc&& d): GenericConvolutionLayer(d), _desc(std::move(d)) {
        _pDesc = &_desc;
    }
    virtual ~SeparableConv2DLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;
    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;

protected:
    SeparableConv2DDesc _desc;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;

public:
    static bool oihw2hwo4i4(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels, int outChannels, int fw, int fh, int unit = 4);
};

} // namespace dp
} // namespace snn
