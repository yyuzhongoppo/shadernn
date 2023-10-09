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

#include "avgpool2d.h"
#include "snn/utils.h"
#include <utility>

namespace snn {
namespace dp { // short for Dynamic Pipeline

class AveragePooling2DLayerGl : public AveragePooling2DLayer {
public:
    AveragePooling2DLayerGl(AveragePooling2DDesc&& d): AveragePooling2DLayer(std::move(d)) {}
    virtual ~AveragePooling2DLayerGl() = default;

protected:
    InferencePassesUptr createFS(const LayerGenOptions&) const override;
    InferencePassesUptr createCS(const LayerGenOptions&) const override;

private:
    void buildPreDefine(std::ostringstream& stream, const LayerGenOptions& options, const std::string& shaderFilePath) const;

    void buildTextureDefLogic(std::ostream& stream, uint32_t inputSliceIndex) const;

    void buildCalcDefLogic(std::ostream& stream) const;

    void buildComputePostDefine(std::ostream& stream, uint32_t outputSliceIndex) const;

    void buildFragPostDefine(std::ostream& stream) const;

    void getPaddingOffset(uint32_t (&offsets)[4]) const;
};

}; // namespace dp
} // namespace snn
