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

public class AlgorithmConfig {

    public enum DenoiserAlgorithm {
        NONE, AIDENOISER, SPATIALDENOISER
    }

    public enum ShaderType {
        COMPUTESHADER, FRAGMENTSHADER
    }

    public enum ClassifierAlgorithm {
        NONE, RESNET18, MOBILENETV2
    }

    public enum DetectionAlgorithm {
        NONE, YOLOV3
    }

    public enum Precision {
        FP32, FP16
    }

    public enum StyleTransfer {
        NONE, CANDY, MOSAIC, POINTILISM, RAIN_PRINCESS, UDNIE
    }

    public int classifierIndex;
    private String[] resnet18Classes = {"None", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    private String[] mobilenetClasses = {"None", "Class 1", "Class 2"};

    private ShaderType denoiserShader;
    private DenoiserAlgorithm denoiserAlgorithm;
    private ClassifierAlgorithm classifierAlgorithm;
    private ShaderType classifierShader;
    private DetectionAlgorithm detectionAlgorithm;
    private ShaderType detectionShader;
    private Precision precision;
    private StyleTransfer styleTransferAlgorithm;

    private static final int NO_CHANGE = 0;
    private static final int CHANGE = 1;
    private static final int CHANGE_PROCESSED = 2;
    private int changeStatus = NO_CHANGE;

    AlgorithmConfig() {
        init();
    }

    private void init() {
        denoiserAlgorithm = DenoiserAlgorithm.NONE;
        denoiserShader = ShaderType.FRAGMENTSHADER;
        classifierAlgorithm = ClassifierAlgorithm.NONE;
        classifierShader = ShaderType.FRAGMENTSHADER;
        detectionAlgorithm = DetectionAlgorithm.NONE;
        detectionShader = ShaderType.FRAGMENTSHADER;
        precision = Precision.FP32;
        styleTransferAlgorithm = StyleTransfer.NONE;
    }

    public String getClassifierOutput() {
        String classifierOutput = "";
        if (this.classifierAlgorithm == ClassifierAlgorithm.RESNET18) {
            classifierOutput = (classifierIndex >= 0 && classifierIndex <= 10) ? resnet18Classes[classifierIndex] : "None";
        } else if (this.classifierAlgorithm == ClassifierAlgorithm.MOBILENETV2) {
            classifierOutput = (classifierIndex >= 0 && classifierIndex <= 2) ? mobilenetClasses[classifierIndex] : "None";
        } else {
            classifierOutput = "N/A";
        }

        return classifierOutput;
    }

    void setDenoiserAlgorithm(DenoiserAlgorithm denoiserAlgorithm) {
        if (this.denoiserAlgorithm != denoiserAlgorithm) {
            changeStatus = CHANGE;
        }
        this.denoiserAlgorithm = denoiserAlgorithm;
    }

    public void setDenoiserShaderType(ShaderType denoiserShader) {
        this.denoiserShader = denoiserShader;
    }

    public boolean isDenoiseSPATIALDENOISER() {return  denoiserAlgorithm == DenoiserAlgorithm.SPATIALDENOISER; }

    public boolean isDenoiseComputeShader() {
        return denoiserShader == ShaderType.COMPUTESHADER;
    }

    void setStyleTransferAlgorithm(StyleTransfer styleTransferAlgorithm) {
        if (this.styleTransferAlgorithm != styleTransferAlgorithm) {
            changeStatus = CHANGE;
        }
        this.styleTransferAlgorithm = styleTransferAlgorithm;
    }
    public boolean isStyleTransferNONE() { return styleTransferAlgorithm == StyleTransfer.NONE; }

    public boolean isStyleTransferCANDY() { return styleTransferAlgorithm == StyleTransfer.CANDY; }

    public boolean isStyleTransferMOSAIC() { return styleTransferAlgorithm == StyleTransfer.MOSAIC; }

    public boolean isStyleTransferPOINTILISM() { return styleTransferAlgorithm == StyleTransfer.POINTILISM; }

    public boolean isStyleTransferRAINPRINCESS() { return styleTransferAlgorithm == StyleTransfer.RAIN_PRINCESS; }

    public boolean isStyleTransferUDNIE() { return  styleTransferAlgorithm == StyleTransfer.UDNIE; }

    public void setClassifierAlgorithm(ClassifierAlgorithm classifierAlgorithm) {
        if (this.classifierAlgorithm != classifierAlgorithm) {
            changeStatus = CHANGE;
        }
        this.classifierAlgorithm = classifierAlgorithm;
        this.classifierIndex = 0;
    }

    public void setClassifierShaderType(ShaderType classifierShader) {
        if (this.classifierShader != classifierShader) {
            changeStatus = CHANGE;
        }
        this.classifierShader = classifierShader;
    }

    public boolean isClassifierResnet18() {
        return this.classifierAlgorithm == ClassifierAlgorithm.RESNET18;
    }

    public boolean isClassifierMobilenetv2() {
        return this.classifierAlgorithm == ClassifierAlgorithm.MOBILENETV2;
    }

    public boolean isClassifierComputeShader() {
        return this.classifierShader == ShaderType.COMPUTESHADER;
    }

    public void setDetectionAlgorithm(DetectionAlgorithm detectionAlgorithm) {
        if (this.detectionAlgorithm != detectionAlgorithm) {
            changeStatus = CHANGE;
        }
        this.detectionAlgorithm = detectionAlgorithm;
    }

    public void setDetectionShaderType(ShaderType detectionShader) {
        if (this.detectionShader != detectionShader) {
            changeStatus = CHANGE;
        }
        this.detectionShader = detectionShader;
    }

    public void setPrecision(Precision precision) {
        if (this.precision != precision) {
            changeStatus = CHANGE;
        }
        this.precision = precision;
    }

    public boolean isDetectionYolov3() {
        return this.detectionAlgorithm == DetectionAlgorithm.YOLOV3;
    }

    public boolean isDetectionComputeShader() {
        return this.detectionShader == ShaderType.COMPUTESHADER;
    }

    public Precision getPrecision() {
        return precision;
    }

    public boolean isFP16() {
        return this.precision == Precision.FP16;
    }

    public boolean isChanged() {
        return this.changeStatus == CHANGE;
    }

    public void setChangeProcessed() {
        if (this.changeStatus == CHANGE) {
            this.changeStatus = CHANGE_PROCESSED;
        }
    }
    public boolean isChangeProcessed() {
        return this.changeStatus == CHANGE_PROCESSED;
    }

    public void resetChange() {
        this.changeStatus = NO_CHANGE;
    }
}