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

struct Conv2DDesc : GenericConvDesc {
    bool useBatchNormalization = false;
    bool useMultiInputs = false;
    std::map<std::string, std::vector<float>> batchNormalization;
    float leakyReluAlpha = 0.0f;
    std::string padding;
    bool useUniformShaders = true;
    std::string paddingT, paddingB, paddingL, paddingR;
    std::string paddingMode = "constant";
    void parse(ModelParser& parser, int layerId);
};

// This is a base class to generates a shader for 2D convolution
class Conv2DLayer : public GenericConvolutionLayer {
public:
    Conv2DLayer(Conv2DDesc&& d) : GenericConvolutionLayer(d), _desc(std::move(d)) {
        _pDesc = &_desc;
    }
    virtual ~Conv2DLayer() = default;
    virtual InferenceGraph::Transform getOutputScaleDimAdjustment() const override;

    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override;

protected:
    Conv2DDesc _desc;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;

public:
    static bool oihw2hwo4i4(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels,
        int outChannels, int fw, int fh, int unit = 4);

    static bool oihw2hwo4i4fp16(const Conv2DSupport::WeightsTensor& inputWeights, std::vector<float>& outVec, int inChannels,
        int outChannels, int fw, int fh, int unit = 4);
};

}; // namespace dp
} // namespace snn
