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

#include "inferencepass.h"
#include "snn/color.h"
#include "conv2dSupport.h"
#include "glUtils.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <variant>

#define MIN_SSBO_BUFFER_LEN_ARM_MALI 4

namespace snn {

// This structure holds information for one OpenGL render pass.
struct InferencePassGl : public InferencePass {
    InferencePassGl() = default;

    SNN_NO_COPY(InferencePassGl);

    InferencePassGl(InferencePassGl&& other) = default;

    struct FsProgram {
        uint32_t outputSliceIndex;
        uint32_t outputSliceCount;
    };
    struct CsProgram {
        // compute shader always bind the texture object as a whole. So there's no need to specify output slice index and count.
        std::string outputImageUniform;
        uint32_t dispatchSize[3];
    };

    // Fragment shader or Compute shader program
    std::variant<FsProgram, CsProgram> program;

    void releaseResources() override {
        InferencePass::releaseResources();
        uptrWeightsConvView.reset();
        // To avoid accidental illegal memory access
        _vecMean = {nullptr, 0};
        _vecVariance = {nullptr, 0};
        _vecBeta = {nullptr, 0};
        _vecGamma = {nullptr, 0};
        weightMeta.clear();
        weightMeta.shrink_to_fit();
        _vecBias = {nullptr, 0};
        uniforms = {};
    }

    // Used in a Fragment Shader only
    std::unique_ptr<Conv2DSupport::WeightsTensorConstView> uptrWeightsConvView;
    
    const Conv2DSupport::WeightsTensorConstView& convWeightsView() const {
        return *uptrWeightsConvView;
    }

    // The following 4 members used in OpenGL in Compute Shader only,
    // in several layers
    // Holds reference to correspondent layers' members
    ArrayCref<float> _vecMean;
    ArrayCref<float> _vecVariance;
    ArrayCref<float> _vecBeta;
    ArrayCref<float> _vecGamma;

    std::vector<uint32_t> weightMeta;

    // Used in Compute Shader only
    // Holds reference to _desc.biases
    ArrayCref<float> _vecBias;

    // Other uniforms. Key is shader variable name.
    std::unordered_map<std::string, gl::SimpleUniform::Value> uniforms;
};

struct InferencePassesGl : public InferencePasses {
    InferencePassesGl() : InferencePasses(GpuBackendType::GL) {}

    static InferencePassesGl* cast(InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::GL);
        return static_cast<InferencePassesGl*>(iPasses);
    }

    static const InferencePassesGl* cast(const InferencePasses* iPasses) {
        SNN_ASSERT(iPasses->backendType == GpuBackendType::GL);
        return static_cast<const InferencePassesGl*>(iPasses);
    }

    std::vector<InferencePassGl> passes;

    InferencePass& operator[](size_t i) override {
        return passes[i];
    }

    const InferencePass& operator[](size_t i) const override {
        return passes[i];
    }
};

}   // namespace snn
