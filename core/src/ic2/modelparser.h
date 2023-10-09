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
#include "conv2dSupport.h"
#include <picojson.h>
#include <string>
#include <fstream>
#include <map>
#include <vector>

namespace snn {
namespace dp {

// This classes can parse a model description, stored in JSON file
// and retrieve its properties
class ModelParser {
private:
    picojson::value _modelOb;
    bool preferHp; // For half precision (16-bit floats)
    bool isBinWeight = false;
    std::ifstream binFile; // For reading weight from separate file
    MRTMode mrtMode;
    WeightAccessMethod weightMode;

public:
    struct CreationParameters {
        const std::string filename;
        bool preferHp;
        MRTMode mrtMode;
        WeightAccessMethod weightMode;
    };

    bool isInputRange01();

    int getLayerCount();

    int getInputHeight();

    int getInputWidth();

    int getInputlayerChannels();

    int getUpscale();

    bool useSubPixel();

    bool normalize();

    bool mergeY2GB();

    bool getPrecision();

    snn::MRTMode getMRTMode();

    snn::WeightAccessMethod getWeightMode();

    std::string getActivation(int layerId);

    int getInputPlanes(int layerId);

    int getOutputPlanes(int layerId);

    picojson::array getWeights(int layerId);

    picojson::array getDepthWiseWeights(int layerId);

    picojson::array getPointwiseWeights(int layerId);

    picojson::array getBias(int layerId);

    std::string getLayerName(int layerId);

    std::string getUseBias(int layerId);

    std::string getPadding(int layerId);

    int getNumInbound(int layerId);

    std::vector<int> getInboundLayerId(int layerId);

    int getInlayerId(int layerId, int inboundNum);

    int getKernelSize(int layerId);

    int getDepthwiseKernelSize(int layerId);

    int getDepthwiseMultiplier(int layerId);

    int getConvolutionLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation, int& kernelSize, int& stride,
                            std::vector<float>& biases, Conv2DSupport::WeightsTensor& weights,
                            bool& _useBatchNormalization,
                            std::map<std::string, std::vector<float>>& batchNormalization, float& leakyReluAlpha, std::string& paddingT, std::string& paddingB,
                            std::string& paddingL, std::string& paddingR, std::string& paddingMode, bool& useMultiInputs);

    int getMaxPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize, int& stride, std::string& paddingMode,
                        std::string& paddingValue, std::string& paddingT, std::string& paddingB, std::string& paddingL, std::string& paddingR);

    int getAddLayer(int& layerID, std::string& activation, float& leakyReluAlpha);

    int getActivationLayer(int& layerID, std::string& activation, float& leakyReluAlpha);

    int getAvgPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize, int& stride, std::string& padding);

    int getAdaptiveAvgPoolLayer(int& layerID, int& numOutputPlanes, int& numInputPlanes, int& poolSize);

    int getInputLayer(int& layerId, uint32_t& inputWidth, uint32_t& inputHeight, uint32_t& inputChannels, uint32_t& inputIndex);

    int getDenseLayer(int& layerID, int& numOutputUnits, int& numInputUnits, std::string& activation, std::vector<std::vector<float>>& weights,
                      std::vector<float>& biases, float& leakyReluAlpha);

    int getFlattenLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation);

    int getYOLOLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes);

    int getDepthwiseConvolutionLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& activation, int& kernelSize, int& stride,
                                     std::vector<float>& biases, Conv2DSupport::WeightsTensor& weights,
                                     bool& useBatchNormalization,
                                     std::map<std::string, std::vector<float>>& batchNormalization, float& leakyReluAlpha, std::string& paddingT,
                                     std::string& paddingB, std::string& paddingL, std::string& paddingR);

    float getUpSamplingScale(int layerId);

    std::string getUpSampling2DInterpolation(int layerId);

    int getBatchNormLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::map<std::string, std::vector<float>>& batchNormalization,
                          std::string& activation, float& leakyReluAlpha);

    int getPaddingLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, std::string& paddingT, std::string& paddingB, std::string& paddingL,
                        std::string& paddingR, std::string& mode, float& constant);

    int getInstanceNormalizationLayer(int& layerId, int& numOutputPlanes, int& numInputPlanes, float& epsilon,
                                      std::map<std::string, std::vector<float>>& batchNormalization, std::string& activation, float& leakyReluAlpha);

    picojson::object getJsonObject(std::string str) { return _modelOb.get(str).get<picojson::object>(); }

    ~ModelParser() {
        if (isBinWeight) {
            binFile.close();
        }
    }

    ModelParser(const CreationParameters cp);
};
} // namespace dp
} // namespace snn
