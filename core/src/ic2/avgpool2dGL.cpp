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
#include "avgpool2dGL.h"
#include "layerFactory.h"
#include "inferencepassGL.h"
#include <string>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <utility>

using namespace snn;
using namespace snn::dp;

static constexpr const char* AVGPOOL2D_FS_ASSET_NAME = "shaders/shadertemplate_fs_avgpooling2d.glsl";
static constexpr const char* AVGPOOL2D_CS_ASSET_NAME = "shaders/3rdparty/shadertemplate_cs_avgpool2d.glsl";

static constexpr const char* INDENT = "    ";

void AveragePooling2DLayerGl::buildPreDefine(std::ostringstream& stream, const GenericModelLayer::LayerGenOptions& options,
                                           const std::string& shaderFilePath) const {
    stream << "#version 320 es\n";
    stream << "// " << shaderFilePath << "\n";
    stream << "#define NUM_INPUT_PLANES " << _desc.numInputPlanes << "\n";
    stream << "#define NUM_OUTPUT_PLANES " << _desc.numOutputPlanes << "\n";
    stream << "#define INPUT_WIDTH " << options.desiredInput[0].width << "\n";
    stream << "#define INPUT_HEIGHT " << options.desiredInput[0].height << "\n";
    stream << "#define NUM_STRIDE " << _desc.stride << "\n";
    stream << "#define N_DIMS " << _desc.stride * _desc.stride << "\n";
    stream << "#define CLAMPED_PADDING\n";

    if (_desc.numInputPlanes <= 4) {
        stream << "#define INPUT_TEXTURE_2D\n";
    }
    stream << std::endl;
}

void AveragePooling2DLayerGl::buildTextureDefLogic(std::ostream& stream, uint32_t inputSliceIndex) const {
    std::vector<float> offsetsW(_desc.kernelSize), offsetsH(_desc.kernelSize);
    int outDim     = ceil((float) inputDims[0].height / (float) _desc.stride);
    float padWidth = (outDim - 1) * _desc.stride + _desc.kernelSize - (float) inputDims[0].height;
    int offSet     = 0;
    if (_desc.padding == "same_upper") {
        offSet = floor(padWidth / 2);
    } else if (_desc.padding == "same_lower") {
        offSet = ceil(padWidth / 2);
    } else if (_desc.padding == "same") {
        if (offSet % 2 != 0) {
            SNN_LOGE("AveragePooling2DLayer::buildTextureDefLogic : Please specify same_upper or same_lower");
        } else {
            offSet = padWidth / 2;
        }
    }
    if (offSet < 0) {
        offSet = 0;
    }
    std::iota(offsetsW.begin(), offsetsW.end(), -offSet - 0.5);
    std::iota(offsetsH.begin(), offsetsH.end(), -offSet - 0.5);

    for (uint32_t i = 0; i < _desc.kernelSize; i++) {
        for (uint32_t j = 0; j < _desc.kernelSize; j++) {
            stream << "\tvec2 texCoord_" << _desc.kernelSize * i + j + 1 << " = (vec2(baseCoord) + ";
            stream << "vec2(" << offsetsH.at(j) << ", " << offsetsW.at(i) << ")) / vec2(maxUV);" << std::endl;
        }
    }

    stream << std::endl;
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        stream << "#if PLANE_COUNT > 3\n";
        stream << "\tint layer3 = " << inputSliceIndex + 3 << ";" << std::endl;
        stream << "\tint layer2 = " << inputSliceIndex + 2 << ";" << std::endl;
        for (uint32_t i = 0; i < _desc.kernelSize; i++) {
            for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                int linearDim = _desc.kernelSize * i + j;
                stream << "FLOAT_PRECISION vec4 t" << linearDim << "_3 = vec4(0.0f, 0.0f, 0.0f, 0.0f);\n";
                stream << "t" << linearDim << "_3 = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "TEXTURE(inputTextures, vec3(texCoord_";
                stream << linearDim + 1 << ", layer3)) : t" << linearDim << "_3;\n";
            }
        }
        for (uint32_t i = 0; i < _desc.kernelSize; i++) {
            for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                int linearDim = _desc.kernelSize * i + j;
                stream << "FLOAT_PRECISION vec4 t" << linearDim << "_2 = vec4(0.0f, 0.0f, 0.0f, 0.0f);\n";
                stream << "t" << linearDim << "_2 = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "TEXTURE(inputTextures, vec3(texCoord_";
                stream << linearDim + 1 << ", layer2)) : t" << linearDim << "_2;\n";
            }
        }
        stream << "#endif\n";
        [[fallthrough]];

    case snn::MRTMode::DOUBLE_PLANE:
        stream << "#if PLANE_COUNT > 1\n";
        stream << "\tint layer1 = " << inputSliceIndex + 1 << ";" << std::endl;
        for (uint32_t i = 0; i < _desc.kernelSize; i++) {
            for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                int linearDim = _desc.kernelSize * i + j;
                stream << "FLOAT_PRECISION vec4 t" << linearDim << "_1 = vec4(0.0f, 0.0f, 0.0f, 0.0f);\n";
                stream << "t" << linearDim << "_1 = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "TEXTURE(inputTextures, vec3(texCoord_";
                stream << linearDim + 1 << ", layer1)) : t" << linearDim << "_1;\n";
            }
        }
        stream << "#endif\n";
        [[fallthrough]];

    case snn::MRTMode::SINGLE_PLANE:
        stream << "\tint layer = " << inputSliceIndex << ";" << std::endl;
        for (uint32_t i = 0; i < _desc.kernelSize; i++) {
            for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                int linearDim = _desc.kernelSize * i + j;
                stream << "FLOAT_PRECISION vec4 t" << linearDim << "_0 = vec4(0.0f, 0.0f, 0.0f, 0.0f);\n";
                stream << "t" << linearDim << "_0 = (checkValid(texCoord_" << linearDim + 1 << ")) ? ";
                stream << "TEXTURE(inputTextures, vec3(texCoord_";
                stream << linearDim + 1 << ", layer)) : t" << linearDim << "_0;\n";
            }
        }
        [[fallthrough]];

    default:
        break;
    }

    stream << std::endl;
}

void AveragePooling2DLayerGl::buildCalcDefLogic(std::ostream& stream) const {
    std::string channels[4]       = {"x", "y", "z", "w"};
    std::string channelsUppers[4] = {"R", "G", "B", "A"};
    std::string planeIds[4]       = {"0", "1", "2", "3"};
    double val                    = 1. / (_desc.kernelSize * _desc.kernelSize);
    stream.precision(4);
    stream << INDENT << "const mediump float val = " << std::fixed << val << ";\n";
    for (auto planeID : planeIds) {
        for (std::size_t idx = 0; idx < 4; idx++) {
            auto currCharUpper = channelsUppers[idx];
            auto curChar       = channels[idx];
            stream << "#ifdef USE_COMPONENT_" << currCharUpper << "_PLANE_" << planeID << "\n";
            if (planeID == "0") {
                for (uint32_t i = 0; i < _desc.kernelSize; i++) {
                    for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                        int linearDim = _desc.kernelSize * i + j;
                        stream << INDENT << "s." << curChar << " += (t" << linearDim << "_" << planeID << "." << curChar << " * val);\n";
                    }
                }
            } else {
                for (uint32_t i = 0; i < _desc.kernelSize; i++) {
                    for (uint32_t j = 0; j < _desc.kernelSize; j++) {
                        int linearDim = _desc.kernelSize * i + j;
                        stream << INDENT << "s" << planeID << "." << curChar << " += (t" << linearDim << "_" << planeID << "." << curChar << " * val);\n";
                    }
                }
            }
            stream << "\n#endif" << std::endl;
        }
    }
}

void AveragePooling2DLayerGl::buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const {
    if (_desc.numOutputPlanes > 4) {
        stream << INDENT << "imageStore(outTexture,ivec3(gl_GlobalInvocationID.xy, " << outputSliceIndex << "), s);\n";
    } else {
        stream << INDENT << "imageStore(outTexture,ivec2(gl_GlobalInvocationID.xy), s);\n";
    }
    stream << "}\n";
}

void AveragePooling2DLayerGl::buildFragPostDefine(std::ostream& stream) const {
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        stream << INDENT << "o_pixel3 = s3;\n";
        stream << INDENT << "o_pixel2 = s2;\n";
        [[fallthrough]];

    case snn::MRTMode::DOUBLE_PLANE:
        stream << INDENT << "o_pixel1 = s1;\n";
        [[fallthrough]];

    default:
        stream << INDENT << "o_pixel = s;\n";
        break;
    }
    stream << "}\n";
}

InferencePassesUptr AveragePooling2DLayerGl::createFS(const GenericModelLayer::LayerGenOptions& options) const {
    std::string shaderTemplateFilePath = AVGPOOL2D_FS_ASSET_NAME;
    std::string fsTemplateCode         = loadShader(shaderTemplateFilePath.c_str());

    int channelsPerPass = 4;
    switch (_desc.mrtMode) {
    case snn::MRTMode::QUAD_PLANE:
        channelsPerPass = 16;
        break;

    case snn::MRTMode::DOUBLE_PLANE:
        channelsPerPass = 8;
        break;

    default:
        break;
    }

    uint32_t numShaderPasses = DIV_AND_ROUND_UP(_desc.numOutputPlanes, channelsPerPass);
    std::ostringstream preDefineStream;
    this->buildPreDefine(preDefineStream, options, shaderTemplateFilePath);
    auto preDefine = preDefineStream.str();

    std::ostringstream postDefineStream;
    this->buildFragPostDefine(postDefineStream);
    auto postDefine = postDefineStream.str();

    std::ostringstream avgPoolLogicStream;
    this->buildCalcDefLogic(avgPoolLogicStream);
    auto avgPoolLogic = avgPoolLogicStream.str();

    InferencePassesUptr ret(new InferencePassesGl());
    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(numShaderPasses);

    for (uint32_t i = 0, j = 0; i < numShaderPasses; i++, j += 4) {
        uint32_t outputChannels =
            static_cast<uint32_t>(std::min(channelsPerPass, static_cast<int>(_desc.numOutputPlanes) - static_cast<int>(i) * channelsPerPass));
        uint32_t planeCount = DIV_4_ROUND_UP(outputChannels);
        std::ostringstream rgbaDefine;
        rgbaDefine << "#define PLANE_COUNT " << planeCount << "\n";
        for (std::size_t planeIdx = 0; planeIdx < planeCount; planeIdx++) {
            std::size_t remainingPlanes = static_cast<uint32_t>(std::min(4, static_cast<int>(outputChannels) - static_cast<int>(planeIdx) * 4));
            switch (remainingPlanes) {
            case 4:
                rgbaDefine << "#define USE_COMPONENT_A_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_B_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 3:
                rgbaDefine << "#define USE_COMPONENT_B_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 2:
                rgbaDefine << "#define USE_COMPONENT_G_PLANE_" << planeIdx << "\n";
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            case 1:
                rgbaDefine << "#define USE_COMPONENT_R_PLANE_" << planeIdx << "\n";
                break;
            default:
                break;
            }
        }

        std::string fsCode = fsTemplateCode;

        std::ostringstream textureDefStream;
        this->buildTextureDefLogic(textureDefStream, DIV_4_ROUND_UP(outputChannels) * i);
        auto textureDef = textureDefStream.str();

        if (_desc.preferHp) {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "mediump");
        } else {
            findAndReplace(fsCode, "_PLACEHOLDER_PRECISION_", "highp");
        }

        findAndReplace(fsCode, "_PLACEHOLDER_TEXTURE_READ_", textureDef);
        findAndReplace(fsCode, "_PLACEHOLDER_CALCULATION_", avgPoolLogic);
        findAndReplace(fsCode, "_PLACEHOLDER_N_DIMS_", std::to_string(outputChannels));
        findAndReplace(fsCode, "_PLACEHOLDER_DEFINES_", "");
        findAndReplace(fsCode, "_PLACEHOLDER_UNIFORMS_DECLARATION_", "");

        InferencePassGl& pass = passes[i];
        pass.source                = preDefine + rgbaDefine.str() + fsCode + postDefine;
        pass.inputs                = {{"inputTextures", 0}}; // Input is currently hard coded in GLSL file.
        pass.program               = InferencePassGl::FsProgram {DIV_4_ROUND_UP(outputChannels) * i, DIV_4_ROUND_UP(outputChannels)};
    }

    return ret;
}

InferencePassesUptr AveragePooling2DLayerGl::createCS(const LayerGenOptions& /*options*/) const {
    InferencePassesUptr ret(new InferencePassesGl());

    std::vector<InferencePassGl>& passes = InferencePassesGl::cast(ret.get())->passes;
    passes.resize(1);

    InferencePassGl& pass = passes[0];

    uint32_t inputWidth  = inputDims[0].width;
    uint32_t inputHeight = inputDims[0].height;
    uint32_t inputDepth  = inputDims[0].depth;

    uint32_t outputWidth  = 0;
    uint32_t outputHeight = 0;
    uint32_t outputDepth  = 0;

    GenericModelLayer::getOutputDims(outputWidth, outputHeight, outputDepth);

    outputDepth = _desc.numOutputPlanes;

    std::string shaderHeader;
    if (_desc.preferHp) {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION mediump\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
                       "#define OUTPUT_FORMAT rgba16f\n";
    } else {
        shaderHeader = "#version 320 es \n"
                       "#define PRECISION highp\n"
                       "precision PRECISION float;\n"
                       "layout(std140) uniform;\n"
                       "#define OUTPUT_FORMAT rgba32f\n";
    }
    if (_desc.numInputPlanes <= 4) {
        shaderHeader += "#define INPUT_TEXTURE_2D\n";
    }
    if (_desc.numOutputPlanes <= 4) {
        shaderHeader += "#define OUTPUT_TEXTURE_2D\n";
    }

    std::string shaderUniforms = "#ifdef INPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2D uInput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=0) readonly uniform PRECISION image2DArray uInput;\n"
                            "#endif\n"
                            "#ifdef OUTPUT_TEXTURE_2D\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2D uOutput;\n"
                            "#else\n"
                            "layout(OUTPUT_FORMAT, binding=3) writeonly uniform PRECISION image2DArray uOutput;\n"
                            "#endif\n";

    std::string shaderMain = loadShader(AVGPOOL2D_CS_ASSET_NAME);

    int kernel    = _desc.kernelSize;
    int stride    = _desc.stride;
    int unit      = 4;
    uint32_t ic_4 = UP_DIV(_desc.numInputPlanes, unit);
    uint32_t oc_4 = UP_DIV(_desc.numOutputPlanes, unit);

    shaderHeader += ("#define WORK_X " + std::to_string(mLocalSize[0]) + "\n");
    shaderHeader += ("#define WORK_Y " + std::to_string(mLocalSize[1]) + "\n");
    shaderHeader += ("#define WORK_Z " + std::to_string(mLocalSize[2]) + "\n");

    uint32_t paddingOffsets[4];
    getPaddingOffsetOrig(paddingOffsets, _desc.padding, _desc.padding, _desc.padding, _desc.padding, kernel);
    SNN_LOGD("Padding: %d, %d, %d, %d", paddingOffsets[0], paddingOffsets[1], paddingOffsets[2], paddingOffsets[3]);
    // Hack it. Looks like not padding on top left in NCNN
    paddingOffsets[0] = 0;
    paddingOffsets[2] = 0;

    pass.uniforms = {{"uPad", glm::ivec2(paddingOffsets[0], paddingOffsets[2])},
                     {"uKernel", glm::ivec2(kernel, kernel)},
                     {"uStride", glm::ivec2(stride, stride)},
                     {"uOutputSize", glm::ivec3(outputWidth, outputHeight, oc_4)},
                     {"uInputSize", glm::ivec3(inputWidth, inputHeight, ic_4)}};
    pass.inputs   = {{"uInput", 0}};
    pass.source   = (shaderHeader + shaderUniforms + shaderMain);
    pass.program  = InferencePassGl::CsProgram {"uOutput",
                                                    // div-by-N is determined by work group size defined CS program.
                                                    {UP_DIV(outputWidth, mLocalSize[0]), UP_DIV(outputHeight, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2])}};

    SNN_LOGV("input:%d:%d:%d, output:%d:%d:%d", inputWidth, inputHeight, inputDepth, outputWidth, outputHeight, outputDepth);

    return ret;
}

void AveragePooling2DLayerGl::getPaddingOffset(uint32_t (&offsets)[4]) const {
    std::string paddingT = this->_desc.paddingT;
    std::string paddingB = this->_desc.paddingB;
    std::string paddingL = this->_desc.paddingL;
    std::string paddingR = this->_desc.paddingR;
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
