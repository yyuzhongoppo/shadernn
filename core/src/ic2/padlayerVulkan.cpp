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
#include "padlayer.h"
#include "layerFactory.h"
#include "inferencepassVulkan.h"
#include <string>
#include <vector>
#include <utility>

DECLARE_LAYER_VULKAN_CLASS(Pad);

using namespace snn;
using namespace snn::dp;

static constexpr const char* PAD_VK_ASSET_NAME = "shaders/shadertemplate_vk_pad.spv";
static constexpr const char* PAD_VK_FP16_ASSET_NAME = "shaders/shadertemplate_vk_pad_fp16.spv";

InferencePassesUptr PadLayerVulkan::createCS(const LayerGenOptions& options) const {
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

    uint32_t padType = 0; // No activation
    if (!_desc.mode.compare("constant")) {
        padType = 0;
    } else if (!_desc.mode.compare("replicate")) {
        padType = 1;
    } else if (!_desc.mode.compare("reflect")) {
        padType = 2;
    }

    int unit      = 4;
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    uint32_t paddingOffsets[4];
    getPaddingOffset(paddingOffsets);
    SNN_LOGD("Padding: %s: %d, %d, %d, %d", _desc.mode.c_str(), paddingOffsets[0], paddingOffsets[1], paddingOffsets[2],
                paddingOffsets[3]);

    std::vector<uint32_t> uniform(12);
    uniform[0] = inputWidth;
    uniform[1] = inputHeight;
    uniform[2] = inputDepth;
    uniform[3] = 1;
    uniform[4] = outputWidth;
    uniform[5] = outputHeight;
    uniform[6] = UP_DIV(outputDepth, 4);
    uniform[7] = 1;
    uniform[8] = padType;
    uniform[9] = padType;
    uniform[10] = paddingOffsets[0];
    uniform[11] = paddingOffsets[2];

    std::pair<std::string, std::vector<uint32_t>> uniformBuffer("2", uniform);
    pass.uniformBuffers.insert(uniformBuffer);

    pass.inputs  = {{"uInput", 0}};

    std::vector<uint8_t> bytes;
    if (_desc.preferHp) {
        bytes = snn::loadEmbeddedAsset(PAD_VK_FP16_ASSET_NAME);
        pass.source = PAD_VK_FP16_ASSET_NAME;
    } else {
        bytes = snn::loadEmbeddedAsset(PAD_VK_ASSET_NAME);
        pass.source = PAD_VK_ASSET_NAME;
    }

    pass.vkCodes.resize((bytes.size() + 3)/4);
    memcpy(pass.vkCodes.data(), bytes.data(), bytes.size());

    pass.program = InferencePassVulkan::VkProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}
