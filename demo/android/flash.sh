# Copyright (C) 2020 - 2022 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
ROOT=`dirname "$(realpath $0)"`

#adb root
#adb remount
#adb shell setenforce 0

clean_phone_json_models()
{
    adb shell rm -rf /data/local/tmp/jsonModel
    adb shell mkdir /data/local/tmp/jsonModel
    adb shell rm -rf /data/local/tmp/files
    adb shell mkdir -p /data/local/tmp/files
    adb shell rm -rf /data/local/tmp/inferenceCoreDump
    adb shell mkdir -p /data/local/tmp/inferenceCoreDump
}

push_phone_json_models()
{   
    clean_phone_json_models
    adb push ${ROOT}/../../modelzoo/ESPCN /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/MobileNetV2 /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/Resnet18 /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/SpatialDenoise /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/U-Net /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/Yolov3-tiny /data/local/tmp/jsonModel
    adb push ${ROOT}/../../modelzoo/StyleTransfer /data/local/tmp/jsonModel
    adb push ${ROOT}/../../core/data/assets /data/local/tmp/files
    adb shell chmod -R 777 /data/local/tmp/files
    adb shell chmod -R 777 /data/local/tmp/jsonModel
}

push_phone_unit_tests()
{
    device_dir="/data/local/tmp/"
    device_assets_dir="/data/local/tmp/files"
    lib_snn="${ROOT}/../../snn-core-install/lib/arm64-v8a/libsnn_core.so"
    libs_opencv_dir="${ROOT}/../../core/3rdparty/opencv/android/lib/arm64-v8a/"
    tests_dir="${ROOT}/../build-test/test/unittest/"
    assets_dir="${ROOT}/../../core/data/assets/"
    adb shell rm -rf ${device_dir}/libsnn_core.so
    adb shell rm -rf ${device_dir}/libopencv*.so
    adb push ${lib_snn} ${device_dir}
    adb push ${libs_opencv_dir}/* ${device_dir}
    adb shell rm -rf ${device_dir}/*Test
    adb push ${tests_dir}/*Test ${device_dir}
    adb push ${assets_dir} ${device_assets_dir}
}

if [ "$1" == "models" ]; then
    push_phone_json_models
elif [ "$1" == "tests" ]; then
    push_phone_unit_tests $1
else 
    ${ROOT}/gradlew installDebug
fi

