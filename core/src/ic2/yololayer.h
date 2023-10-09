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

#include <snn/snn.h>
#include <snn/utils.h>
#include <snn/imageTexture.h>
#include "snn/inferencegraph.h"
#include "inferencepass.h"
#include "modelparser.h"
#include <utility>
#include <set>
#include <string>

#include "genericlayer.h"

namespace snn {
namespace dp { // short for Dynamic Pipeline
struct YOLODesc : CommonLayerDesc {
    void parse(ModelParser& parser, int layerId);
};

// This class implements functionality of the final layer of YOLO model. It is processed on CPU.
class YOLOLayer : public GenericModelLayer {
public:
    YOLOLayer(YOLODesc&& d): GenericModelLayer(d), _yoloDesc(std::move(d)) {}
    YOLOLayer(const YOLOLayer& d) = delete;
    YOLOLayer& operator=(const YOLOLayer& d) = delete;
    virtual ~YOLOLayer() = default;

    InferenceGraph::Transform getOutputScaleDimAdjustment() const override { return {0, {{1.0f, 1.0f, 0.0f, 0.0f}}}; };

    virtual void getOutputDims(uint32_t& width, uint32_t& height, uint32_t& depth) const override {
        width  = 100 * 6; // Max 100 bounding box
        height = 1;
        depth  = 1;
    }

    virtual void computeImageTexture(ImageTextureArray& inputMat, ImageTextureArray& outputMat) override;

    virtual snn::InferenceGraph::LayerExecutionType getLayerExecutionType() const override { return executeBackend; }
    virtual void setLayerExecutionType(InferenceGraph::LayerExecutionType newExecution) override { executeBackend = newExecution; }

    virtual void createInferencePasses(const LayerGenOptions& /*options*/) override {}

    virtual bool isTransition() const override { return true; }

private:
    YOLODesc _yoloDesc;
    snn::InferenceGraph::LayerExecutionType executeBackend = snn::InferenceGraph::LayerExecutionType::CPU;
};

}; // namespace dp
} // namespace snn
