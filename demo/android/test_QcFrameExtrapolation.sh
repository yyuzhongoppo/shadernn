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

if [ "$1" == "--help" ]; then
    echo "Usage:"
    echo "adbTest.sh {test} {params...}"
    exit 1
fi

device_dir=/data/local/tmp/
tests_dir="${ROOT}/../build-test/test/unittest/"
adb push $tests_dir/qcFrameExtrapolationTest ${device_dir}
adb shell "rm -rf ${device_dir}/$1"
adb push $1 ${device_dir}
adb shell "mkdir ${device_dir}/$1/extrapolated"
adb shell "mkdir ${device_dir}/$1/gsr"
adb shell "mkdir ${device_dir}/$1/fsr"
adb shell "mkdir ${device_dir}/$1/ofhs"
adb shell "mkdir ${device_dir}/$1/warp"
adb shell "mkdir ${device_dir}/$1/motion"
adb shell "mkdir ${device_dir}/$1/motion_synth"
adb shell "cd ${device_dir} && LD_LIBRARY_PATH=. ./qcFrameExtrapolationTest $1 --test $2 --scale_factor $3 --offset $4"
adb pull ${device_dir}/$1/extrapolated .
adb pull ${device_dir}/$1/gsr .
adb pull ${device_dir}/$1/fsr .
adb pull ${device_dir}/$1/ofhs .
adb pull ${device_dir}/$1/warp .
adb pull ${device_dir}/$1/motion .
adb pull ${device_dir}/$1/motion_synth .
adb shell "rm -rf ${device_dir}/$1"