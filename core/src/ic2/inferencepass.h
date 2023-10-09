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

#include "snn/snn.h"
#include "snn/color.h"
#include "snn/utils.h"
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <optional>
#include <utility>

namespace snn {

// This structure holds information for one render pass.
struct InferencePass {
    InferencePass() = default;

    SNN_NO_COPY(InferencePass);

    InferencePass(InferencePass&& other) = default;

    virtual ~InferencePass() = default;

    // Release resources after they were copied to GPU / RenderPass members
    // TODO: remove runtime data and delete the whole object
    virtual void releaseResources() {
        weightDims = {};
        _vecWeights.clear();
        _vecWeights.shrink_to_fit();
    }

    // shader source code
    std::string source;

    // key is shader variable name. value is index into the layer's input buffer array.
    std::unordered_map<std::string, uint32_t> inputs; // input buffer uniforms.

    // Other uniforms. Key is shader variable name.
    std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> runtimeUniforms; // name and offset/size of runtime parameter
    std::vector<uint32_t> runtimeData; // run tcv::ime data in a period
    uint32_t period; // loop every period * 4 in runtime data
    uint32_t passId = 0; // nth of the render pass
    uint32_t totalPasses = 1; // total render passes needed for this operator

    uint32_t inputHeight, inputWidth, inputChannels;

    // Weights dimensions
    // TODO: remove map, because it contain only one element
    std::map<std::string, std::array<uint32_t, 3>> weightDims;

    // Used in Compute Shader and Vulkan Shader only
    // only in Conv2D, DepthwiseConv2D and Dense layers
    // Stores weights in hwo4i4 format, where 4 means 4 alignment
    // Copied from GenericConvDesc::weightsConv()
    std::vector<float> _vecWeights;
};

struct InferencePasses {
public:
    virtual ~InferencePasses() = default;

protected:
    InferencePasses(GpuBackendType backendType_)
        : backendType(backendType_)
    {}

public:
    const GpuBackendType backendType;

    virtual InferencePass& operator[](size_t i) = 0;
    virtual const InferencePass& operator[](size_t i) const = 0;
};

typedef std::unique_ptr<InferencePasses> InferencePassesUptr;

}   // namespace snn
