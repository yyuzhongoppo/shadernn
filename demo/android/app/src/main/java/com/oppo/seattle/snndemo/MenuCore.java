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
package com.oppo.seattle.snndemo;

import android.content.Context;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

class MenuCore {
    private static final String TAG = "MenuCore";
    private Context mContext;
    private Menu mMenu;
    private AlgorithmConfig mAC;

    MenuCore(Context context, Menu menu, AlgorithmConfig ac) {
        mContext = context;
        mMenu = menu;
        mAC = ac;
    }

    private boolean isOptionsItemModelRun(MenuItem item) {
        if (item.getItemId() == R.id.fp16 || item.getItemId() == R.id.fp32
                || item.getItemId() == R.id.compute_shader || item.getItemId() == R.id.fragment_shader) {
            return false;
        }
        return true;
    }

    boolean onOptionsItemSelected(MenuItem item) {
        item.setChecked(true);
        // Expand submodels
        if (mMenu.findItem(R.id.classifier).isChecked()) {
            mMenu.setGroupVisible(R.id.classifier_choices, true);
        } else {
            mMenu.setGroupVisible(R.id.classifier_choices, false);
        }
        if (mMenu.findItem(R.id.style_transfer).isChecked()) {
            mMenu.setGroupVisible(R.id.style_transfer_choices, true);
        } else {
            mMenu.setGroupVisible(R.id.style_transfer_choices, false);
        }
        boolean concreteModelSelected = false;
        if (mMenu.findItem(R.id.spatialdenoise).isChecked() ||
                mMenu.findItem(R.id.classifier).isChecked() && (
                        mMenu.findItem(R.id.resnet18_classifier).isChecked() ||
                                mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()
                ) ||
                mMenu.findItem(R.id.yolov3_detection).isChecked() ||
                mMenu.findItem(R.id.style_transfer).isChecked() && (
                        mMenu.findItem(R.id.style_candy).isChecked() ||
                                mMenu.findItem(R.id.style_mosaic).isChecked() ||
                                mMenu.findItem(R.id.style_pointilism).isChecked() ||
                                mMenu.findItem(R.id.style_rain_princess).isChecked() ||
                                mMenu.findItem(R.id.style_udnie).isChecked()
                )
            ) {
            concreteModelSelected = true;
        }

        // If shader choice is available (OpenGL) ?
        if (mMenu.findItem(R.id.compute_shader).isVisible() && mMenu.findItem(R.id.fragment_shader).isVisible()) {
            // Disable fragment shader choice for some models, because compiling shaders is too slow
            if (mMenu.findItem(R.id.classifier).isChecked() && mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()) {
                mMenu.findItem(R.id.fragment_shader).setEnabled(false);
                mMenu.findItem(R.id.compute_shader).setChecked(true);
            } else {
                mMenu.findItem(R.id.fragment_shader).setEnabled(true);
            }
            // Set up default shader type
            if (concreteModelSelected) {
                if (!mMenu.findItem(R.id.compute_shader).isChecked() && !mMenu.findItem(R.id.fragment_shader).isChecked()) {
                    if (mMenu.findItem(R.id.spatialdenoise).isChecked()) {
                        // Fragment shader is default for spatial denoise
                        mMenu.findItem(R.id.fragment_shader).setChecked(true);
                    } else {
                        // Compute shader is default for all other models
                        mMenu.findItem(R.id.compute_shader).setChecked(true);
                    }
                }
            }
        }

        if (concreteModelSelected) {
            mMenu.findItem(R.id.modelRunId).setEnabled(true);
        } else {
            mMenu.findItem(R.id.modelRunId).setEnabled(false);
        }
        if (item.getItemId() == R.id.modelRunId) {
            // Set state, close menu, run model
            setState();
            return true;
        } else {
            keepMenuOpen(item);
            return false;
        }
    }

    private void setState() {
        boolean computeShader = mMenu.findItem(R.id.compute_shader).isChecked();
        if (mMenu.findItem(R.id.spatialdenoise).isChecked()) {
            mAC.setDenoiserAlgorithm(AlgorithmConfig.DenoiserAlgorithm.SPATIALDENOISER);
            if (computeShader) {
                mAC.setDenoiserShaderType(AlgorithmConfig.ShaderType.COMPUTESHADER);
            } else {
                mAC.setDenoiserShaderType(AlgorithmConfig.ShaderType.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.resnet18_classifier).isChecked()) {
            mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.RESNET18);
            if (computeShader) {
                mAC.setClassifierShaderType(AlgorithmConfig.ShaderType.COMPUTESHADER);
            } else {
                mAC.setClassifierShaderType(AlgorithmConfig.ShaderType.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.mobilenetv2_classifier).isChecked()) {
            mAC.setClassifierAlgorithm(AlgorithmConfig.ClassifierAlgorithm.MOBILENETV2);
            if (computeShader) {
                mAC.setClassifierShaderType(AlgorithmConfig.ShaderType.COMPUTESHADER);
            } else {
                mAC.setClassifierShaderType(AlgorithmConfig.ShaderType.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.yolov3_detection).isChecked()) {
            mAC.setDetectionAlgorithm(AlgorithmConfig.DetectionAlgorithm.YOLOV3);
            if (computeShader) {
                mAC.setDetectionShaderType(AlgorithmConfig.ShaderType.COMPUTESHADER);
            } else {
                mAC.setDetectionShaderType(AlgorithmConfig.ShaderType.FRAGMENTSHADER);
            }
        } else if (mMenu.findItem(R.id.style_candy).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.CANDY);
        } else if (mMenu.findItem(R.id.style_mosaic).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.MOSAIC);
        } else if (mMenu.findItem(R.id.style_pointilism).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.POINTILISM);
        } else if (mMenu.findItem(R.id.style_rain_princess).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.RAIN_PRINCESS);
        } else if (mMenu.findItem(R.id.style_udnie).isChecked()) {
            mAC.setStyleTransferAlgorithm(AlgorithmConfig.StyleTransfer.UDNIE);
        }
        if (mMenu.findItem(R.id.fp32).isChecked()) {
            mAC.setPrecision(AlgorithmConfig.Precision.FP32);
        } else {
            mAC.setPrecision(AlgorithmConfig.Precision.FP16);
        }
    }

    private void keepMenuOpen(MenuItem item) {
        item.setShowAsAction(MenuItem.SHOW_AS_ACTION_COLLAPSE_ACTION_VIEW);
        item.setActionView(new View(mContext));
        item.setOnActionExpandListener(new MenuItem.OnActionExpandListener(){
            @Override
            public boolean onMenuItemActionExpand(MenuItem item){
                return false;
            }

            @Override
            public boolean onMenuItemActionCollapse(MenuItem item){
                return false;
            }
        });
    }
}
