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

#define PROFILING_CPU 1

#include "snn/mdArray.h"
#include "snn/mdArrayImpl.h"
#include "snn/utils.h"
#include "snn/image.h"
#include "ic2/conv2d.h"
#include "conv2dSupport.h"
#include <stdio.h>
#include <cstdlib>
#include <ctime>

int main(int , char **) {
    snn::MdArray<int, 3> arr({2, 3, 1});
    arr[0][0][0] = 1;
    arr[0][1][0] = 2;
    arr[0][2][0] = 3;
    arr[1][0][0] = 4;
    arr[1][1][0] = 5;
    arr[1][2][0] = 6;
    SNN_CHK(arr.data()[0] == 1);
    SNN_CHK(arr.data()[1] == 2);
    SNN_CHK(arr.data()[2] == 3);
    SNN_CHK(arr.data()[3] == 4);
    SNN_CHK(arr.data()[4] == 5);
    SNN_CHK(arr.data()[5] == 6);

    SNN_CHK((arr[{1,1,0}] == 5));

    const auto& arr1 = arr;
    SNN_CHK(arr1[1][2][0] == 6);
    (void)arr1;

    snn::MdArray<int, 1> arr2(2);
    arr2[0] = 5;

    snn::MdArrayView<int, 3> arrView = arr.view().slice({{0, 1}, {1, {}}});
    (void) arrView;
    SNN_CHK(arrView.dim(0) == 1);
    SNN_CHK(arrView.dim(1) == 2);
    SNN_CHK(arrView.dim(2) == 1);
    SNN_CHK(arrView[0][1][0] == 3);

    int rawArr[] = {1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400};
    snn::MdArrayView<int, 3> arrView1(rawArr, sizeof(rawArr) / sizeof(int), {3, 2, 2});
    SNN_CHK(arrView1[1][1][1] == 40);
    SNN_CHK((arrView1[2][0][0] == 100));

    snn::MdArray<int, 3> arrPerm1 = arr.permute({2, 1, 0});
    arrPerm1.print();

    snn::MdArray<int, 3> arrPerm2 = arr.permute({1, 0, 2});
    arrPerm2.print();

    snn::Conv2DSupport::WeightsTensor w({512, 512, 5, 5});
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < w.size(); ++i) {
        *(w.data() + i) = (float)std::rand() / RAND_MAX;
    }
    std::vector<float> v(w.size());
    {
        PROFILE_TIME_AND_LOG(oihw2hwo4i4, "oihw2hwo4i4");
        snn::dp::Conv2DLayer::oihw2hwo4i4(w, v, w.dim(1), w.dim(0), w.dim(3), w.dim(2));
    }
    std::vector<float> v1(w.size());
    {
        PROFILE_TIME_AND_LOG(permute, "permute");
        auto w1 = w.reshape<5>({w.dim(0) / 4, 4, w.dim(1), w.dim(2), w.dim(3)});
        w1.permute({3, 4, 0, 2, 1}, v1.data(), v1.size());
    }
    SNN_CHK(v == v1);

    std::vector<float> vh(w.size());
    {
        PROFILE_TIME_AND_LOG(oihw2hwo4i4fp16, "oihw2hwo4i4fp16");
        snn::dp::Conv2DLayer::oihw2hwo4i4fp16(w, vh, w.dim(1), w.dim(0), w.dim(3), w.dim(2));
    }
    std::vector<snn::FP16> vh1(w.size());
    {
        PROFILE_TIME_AND_LOG(permuteFP16, "permuteFP16");
        auto w1 = w.reshape<5>({w.dim(0) / 4, 4, w.dim(1), w.dim(2), w.dim(3)});
        w1.template permute<snn::FP16>({3, 4, 0, 2, 1}, vh1.data(), vh1.size());
    }
    SNN_CHK(memcmp(vh.data(), vh1.data(), vh1.size() * sizeof(snn::FP16)) == 0);
    
    // Testing time of [] operator
    float sum = 0.0f;
    float sum1;
    {
        PROFILE_TIME_AND_LOG(total_sum_flat, "total_sum_flat");
        for (uint32_t i = 0, m = 0; i < w.dim(0); ++i)
            for (uint32_t j = 0; j < w.dim(1); ++j)
                for (uint32_t k = 0; k < w.dim(2); ++k)
                    for (uint32_t l = 0; l < w.dim(3); ++l, ++m)
                        sum += w.data()[m];
    }
    sum1 = 0.0f;
    {
        PROFILE_TIME_AND_LOG(total_sum_md, "total_sum_md");
        for (uint32_t i = 0; i < w.dim(0); ++i) {
            const auto& w0 = w[i];
            for (uint32_t j = 0; j < w0.dim(0); ++j) {
                const auto& w1 = w0[j];
                for (uint32_t k = 0; k < w1.dim(0); ++k) {
                    const auto& w2 = w1[k];
                    for (uint32_t l = 0; l < w2.dim(0); ++l) {
                        sum1 += w2[l];
                    }
                }
            }
        }
    }
    SNN_CHK(sum1 == sum);
    sum1 = 0.0f;
    {
        PROFILE_TIME_AND_LOG(total_sum_arr_index, "total_sum_arr_index");
        for (uint32_t i = 0; i < w.dim(0); ++i)
            for (uint32_t j = 0; j < w.dim(1); ++j)
                for (uint32_t k = 0; k < w.dim(2); ++k)
                    for (uint32_t l = 0; l < w.dim(3); ++l)
                        sum1 += w[{i, j, k, l}];
    }
    SNN_CHK(sum1 == sum);
    sum1 = 0.0f;
    auto wv = w.view();
    {
        PROFILE_TIME_AND_LOG(total_sum_view_md, "total_sum_view_md");
        for (uint32_t i = 0; i < wv.dim(0); ++i) {
            const auto& w0 = wv[i];
            for (uint32_t j = 0; j < w0.dim(0); ++j) {
                const auto& w1 = w0[j];
                for (uint32_t k = 0; k < w1.dim(0); ++k) {
                    const auto& w2 = w1[k];
                    for (uint32_t l = 0; l < w2.dim(0); ++l) {
                        sum1 += w2[l];
                    }
                }
            }
        }
    }
    SNN_CHK(sum1 == sum);
 
    printf("Done\n");
    return 0;
}