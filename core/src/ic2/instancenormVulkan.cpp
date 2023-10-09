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
#include "instancenorm.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include "uvkc/vulkan/pipeline.h"
#include <string>
#include <cstring>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(InstanceNorm);

using namespace snn;
using namespace snn::dp;

static constexpr const char* INSTANCENORM_VK_ASSET_NAME = "shaders/shadertemplate_vk_instancenorm.spv";
static constexpr const char* INSTANCENORM_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_instancenorm_fp16.spv";

InferencePassesUptr InstanceNormLayerVulkan::createCS(const LayerGenOptions& options) const {
    (void) options;

    InferencePassesUptr ret(new InferencePassesVulkan());

    std::vector<InferencePassVulkan>& passes = InferencePassesVulkan::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassVulkan& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    uint32_t activation = 0; // No activation
    float leakyValue = _desc.leakyReluAlpha;
    if (!_desc.activation.compare("relu") || !_desc.activation.compare("Relu")) {
        activation = 1;
    } else if (!_desc.activation.compare("relu6")) {
        activation = 2;
    } else if (!_desc.activation.compare("tanh")) {
        activation = 3;
    } else if (!_desc.activation.compare("sigmoid")) {
        activation = 4;
    } else if (!_desc.activation.compare("leakyRelu")) {
        activation = 5;
    } else if (!_desc.activation.compare("SiLU")) {
        activation = 6;
    }

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t maxThreads   = 256;
    uint32_t threadWidth  = std::min(maxThreads, outputWidth);
    uint32_t threadHeight = std::min(maxThreads / threadWidth, outputHeight);
    std::vector<uint32_t> localSize {threadWidth, threadHeight, 1};

    std::vector<uvkc::vulkan::Pipeline::SpecConstant> specConstants = {
        {0, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = localSize[0]}},
        {1, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = localSize[1]}},
        {2, uvkc::vulkan::Pipeline::SpecConstant::Type::u32, { .u32 = localSize[2]}},
    };
    pass.specConstants = specConstants;

    const std::vector<float>& beta = _desc.instanceNormalization.at("beta");
    pass.objectBuffers["3"] = {beta.data(), beta.size()};

    const std::vector<float>& gamma = _desc.instanceNormalization.at("gamma");
    pass.objectBuffers["4"] = {gamma.data(), gamma.size()};

    std::vector<uint32_t> uniform(8);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = localSize[0];
    uniform[5] = localSize[1];
    uniform[6] = activation;
    unsigned char* target = (unsigned char*)uniform.data() + 7 * sizeof(uint32_t);
    std::memcpy(target, &leakyValue, sizeof(uint32_t));

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uint8_t> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(INSTANCENORM_VK_FP16_ASSET_NAME);
    } else {
        bytes = snn::loadEmbeddedAsset(INSTANCENORM_VK_ASSET_NAME);
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {1, 1, UP_DIV(oc_4, localSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
