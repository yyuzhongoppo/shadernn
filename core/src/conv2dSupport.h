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

#include <snn/mdArray.h>
#include <vector>
#include <optional>

namespace snn {

namespace Conv2DSupport {

typedef MdArray<float, 4> WeightsTensor;

typedef MdArrayView<float, 4> WeightsTensorView;

typedef MdArrayView<const float, 4> WeightsTensorConstView;

}

} // namespace snn