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

#include "calculation.h"
#include "genericlayer.h"
#include "snn/snn.h"
#include "snn/utils.h"
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class CalculateLayerGl : public ShaderLayer {
public:
    CalculateLayerGl(CalculateDesc&& d): ShaderLayer(std::move(d)) {}
    virtual ~CalculateLayerGl() = default;

protected:
    InferencePassesUptr createFS(const LayerGenOptions&) const override;
    InferencePassesUptr createCS(const LayerGenOptions&) const override {
        SNN_LOGW("Compute shader not implemented! Falling back to fragment shader.");
        return {};
    }
};

}; // namespace dp
} // namespace snn
