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
#include "snn/contextFactory.h"
#include "snn/utils.h"
#include "snn/imageTextureFactory.h"
#include "snn/image.h"
#include "snn/deviceTimer.h"

#include "testutil.h"
#ifdef SUPPORT_GL
#include "imageTextureGL.h"
#include <glad/glad.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif
#include <string>
#include <algorithm>
#include <dirent.h>
#include <stdio.h>

#ifdef Success
#undef Success
#endif
#include "CLI/CLI.hpp"

const char* const QCOM_F_EXTRAPOLATION_EXT_NAME = "GL_QCOM_frame_extrapolation";
const char* const QCOM_M_ESTIMATION_EXT_NAME = "GL_QCOM_motion_estimation";
const char* const F_EXTRAPOLATION_FUN_NAME = "glExtrapolateTex2DQCOM";
const char* const F_M_ESTIMATION_FUN_NAME = "glTexEstimateMotionQCOM";
typedef void (*P_EXTRAPOLATE_TEX)(uint, uint, uint, float);
typedef void (*P_ESTIMATE_M)(uint, uint, uint);

#define MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM 0x8C90
#define MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM 0x8C91

struct FileNameToSort
{
    FileNameToSort(const std::string& name)
        : origName(name)
    {
        if (name.length() < 10 ) {
            paddedName = snn::formatString("%.*s%s", std::max(0, 6 - (int)name.size()), "000000", name.c_str());
        } else { 
            paddedName = name;
        }
    }

    std::string origName;
    std::string paddedName;
};

static std::string extrapolName(const std::string& name1, const std::string& name2, float scaleFactor) {
    if (scaleFactor < 0.0f) {
        return name1.substr(0, name1.size()-4) + "_" + std::to_string(std::abs(scaleFactor)) + name2;
    } else {
        return name2 + std::to_string(scaleFactor) + ".png";
    }
}

static std::string motionEstName(const std::string& name1, const std::string& name2) {
    return name1 + "->-" + name2;
}

static float rgba2Y(float r, float g, float b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

static snn::ManagedImage<snn::R8> rgba2Y(const snn::TypedImage<snn::Rgba8>& src) {
    snn::ManagedImage<snn::R8> target({snn::ColorFormat::R8, src.width(), src.height(), 1, 1});
    for (uint32_t h = 0; h < src.height(); h++) {
        for (uint32_t w = 0; w < src.width(); w++) {
            const auto& pixel = src.at(0, w, h, 0);
            target.at(0, w, h, 0).u8 = (uint8_t)rgba2Y(pixel.r, pixel.g, pixel.b);
        }
    }
    return target;
}

static void rgba2Y(snn::ImageTexture& src, snn::ImageTexture& target) {
    SNN_ASSERT(src.format() == snn::ColorFormat::RGB8 || src.format() == snn::ColorFormat::RGBA8);
    SNN_ASSERT(target.format() == snn::ColorFormat::R8);
    SNN_ASSERT(target.getDims() == src.getDims());
    for (uint32_t h = 0; h < src.height(); h++) {
        for (uint32_t w = 0; w < src.width(); w++) {
            uint8_t* p = src.at(0, w, h, 0);
            *target.at(0, w, h, 0) = (uint8_t)rgba2Y(p[0], p[1], p[2]);
        }
    }
}

static void offsetTex(snn::ImageTexture& src, snn::ImageTexture& target, int offset) {
    SNN_ASSERT(src.format() == snn::ColorFormat::RGB8 || src.format() == snn::ColorFormat::RGBA8);
    SNN_ASSERT(target.format() == src.format());
    SNN_ASSERT(target.getDims() == src.getDims());
    size_t bytes = snn::getColorFormatDesc(src.format()).bytes();
    for (uint32_t h = 0; h < src.height(); h++) {
        for (int w = 0; w < src.width(); w++) {
            int wTarget = w + offset;
            if (wTarget >= 0 && wTarget < target.width()) {
                uint8_t* pSrc = src.at(0, w, h, 0);
                uint8_t* pTarget = target.at(0, wTarget, h, 0);
                memcpy(pTarget, pSrc, bytes);
            }
        }
    }
}

static void rgba2R(snn::ImageTexture& src, snn::ImageTexture& target, float factor = 1.0f, float* pMax = nullptr) {
    SNN_ASSERT(src.format() == snn::ColorFormat::RGBA32F);
    SNN_ASSERT(target.format() == snn::ColorFormat::R32F);
    SNN_ASSERT(target.getDims() == src.getDims());
    for (uint32_t h = 0; h < src.height(); h++) {
        for (uint32_t w = 0; w < src.width(); w++) {
            float* pSrc = (float*)src.at(0, w, h, 0);
            float* pTarget = (float*)target.at(0, w, h, 0);
            //*pTarget = std::sqrt(pSrc[0] * pSrc[0] + pSrc[1] * pSrc[1]) * factor; // Copying R + G channels
            *pTarget = pSrc[0]*factor; // Copying R + G channels
            if (pMax) {
                *pMax = std::max(*pMax, std::abs(pSrc[0])); // R channel
                *pMax = std::max(*pMax, std::abs(pSrc[1])); // G channel
            }
        }
    }
}

#define PROFILING 1
typedef uint32_t AU1;

#define STB_IMAGE_IMPLEMENTATION
#include "src/image_utils.h"

#define A_CPU
#include "src/ffx_a.h"
#include "src/ffx_fsr1.h"

static void runFSR(struct FSRConstants fsrData, uint32_t fsrProgramEASU, uint32_t fsrProgramRCAS, uint32_t fsrData_vbo, uint32_t inputImage, uint32_t outputImage) {
    uint32_t displayWidth = fsrData.output.width;
    uint32_t displayHeight = fsrData.output.height;

    static const int threadGroupWorkRegionDim = 16;
    int dispatchX = (displayWidth + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    int dispatchY = (displayHeight + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;


    // binding point constants in the shaders
    const int inFSRDataPos = 0;
    const int inFSRInputTexture = 1;
    const int inFSROutputTexture = 2;

    { // run FSR EASU
        glUseProgram(fsrProgramEASU);

        // connect the input uniform data
        glBindBufferBase(GL_UNIFORM_BUFFER, inFSRDataPos, fsrData_vbo);

        // bind the input image to a texture unit
        glActiveTexture(GL_TEXTURE0 + inFSRInputTexture);
        glBindTexture(GL_TEXTURE_2D, inputImage);

        // connect the output image
        glBindImageTexture(inFSROutputTexture, outputImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(dispatchX, dispatchY, 1);
        glFinish();
    }

    {
        // FSR RCAS
        // connect the input uniform data
        glBindBufferBase(GL_UNIFORM_BUFFER, inFSRDataPos, fsrData_vbo);

        // connect the previous image's output as input
        glActiveTexture(GL_TEXTURE0 + inFSRInputTexture);
        glBindTexture(GL_TEXTURE_2D, outputImage);

        // connect the output image which is the same as the input image
        glBindImageTexture(inFSROutputTexture, outputImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glUseProgram(fsrProgramRCAS);
        glDispatchCompute(dispatchX, dispatchY, 1);
        glFinish();
    }
}

static void runBilinear(struct FSRConstants fsrData, uint32_t bilinearProgram, int32_t fsrData_vbo, uint32_t inputImage, uint32_t outputImage) {
    uint32_t displayWidth = fsrData.output.width;
    uint32_t displayHeight = fsrData.output.height;

    static const int threadGroupWorkRegionDim = 16;
    int dispatchX = (displayWidth + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    int dispatchY = (displayHeight + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;


    // binding point constants in the shaders
    const int inFSRDataPos = 0;
    const int inFSRInputTexture = 1;
    const int inFSROutputTexture = 2;

    { // run FSR EASU
        glUseProgram(bilinearProgram);

        // connect the input uniform data
        glBindBufferBase(GL_UNIFORM_BUFFER, inFSRDataPos, fsrData_vbo);

        // bind the input image to a texture unit
        glActiveTexture(GL_TEXTURE0 + inFSRInputTexture);
        glBindTexture(GL_TEXTURE_2D, inputImage);

        // connect the output image
        glBindImageTexture(inFSROutputTexture, outputImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(dispatchX, dispatchY, 1);
        glFinish();
    }
}

uint32_t createOutputImage(struct FSRConstants fsrData) {
    uint32_t outputImage = 0;
    glGenTextures(1, &outputImage);
    glBindTexture(GL_TEXTURE_2D, outputImage);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, fsrData.output.width, fsrData.output.height);
    glBindTexture(GL_TEXTURE_2D, 0);

    return outputImage;
}

#define VERTEX_POS_INDX  0
#define TEXTURE_POS_INDX 1
#define LOGCATE printf
#define GO_CHECK_GL_ERROR(...)   LOGCATE("CHECK_GL_ERROR %s glGetError = %d, line = %d, \n",  __FUNCTION__, glGetError(), __LINE__)

void CheckGLError(const char *pGLOperation)
{
    for (GLint error = glGetError(); error; error = glGetError())
    {
        LOGCATE("GLUtils::CheckGLError GL Operation %s() glError (0x%x)\n", pGLOperation, error);
    }

}

GLuint LoadShader(GLenum shaderType, const char *pSource)
{
    GLuint shader = 0;
	// FUN_BEGIN_TIME("GLUtils::LoadShader")
        shader = glCreateShader(shaderType);
        if (shader)
        {
            glShaderSource(shader, 1, &pSource, NULL);
            glCompileShader(shader);
            GLint compiled = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
            if (!compiled)
            {
                GLint infoLen = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
                if (infoLen)
                {
                    char* buf = (char*) malloc((size_t)infoLen);
                    if (buf)
                    {
                        glGetShaderInfoLog(shader, infoLen, NULL, buf);
                        LOGCATE("GLUtils::LoadShader Could not compile shader %d:\n%s\n", shaderType, buf);
                        free(buf);
                    }
                    glDeleteShader(shader);
                    shader = 0;
                }
            }
        }
	// FUN_END_TIME("GLUtils::LoadShader")
	return shader;
}

GLuint CreateProgram(const char *pVertexShaderSource, const char *pFragShaderSource, GLuint &vertexShaderHandle, GLuint &fragShaderHandle)
{
    GLuint program = 0;
    // FUN_BEGIN_TIME("GLUtils::CreateProgram")
        vertexShaderHandle = LoadShader(GL_VERTEX_SHADER, pVertexShaderSource);
        if (!vertexShaderHandle) return program;
        fragShaderHandle = LoadShader(GL_FRAGMENT_SHADER, pFragShaderSource);
        if (!fragShaderHandle) return program;

        program = glCreateProgram();
        if (program)
        {
            glAttachShader(program, vertexShaderHandle);
            CheckGLError("glAttachShader");
            glAttachShader(program, fragShaderHandle);
            CheckGLError("glAttachShader");
            glLinkProgram(program);
            GLint linkStatus = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

            glDetachShader(program, vertexShaderHandle);
            glDeleteShader(vertexShaderHandle);
            vertexShaderHandle = 0;
            glDetachShader(program, fragShaderHandle);
            glDeleteShader(fragShaderHandle);
            fragShaderHandle = 0;
            if (linkStatus != GL_TRUE)
            {
                GLint bufLength = 0;
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
                if (bufLength)
                {
                    char* buf = (char*) malloc((size_t)bufLength);
                    if (buf)
                    {
                        glGetProgramInfoLog(program, bufLength, NULL, buf);
                        // LOGCATE("GLUtils::CreateProgram Could not link program:\n%s\n", buf);
                        free(buf);
                    }
                }
                glDeleteProgram(program);
                program = 0;
            }
        }
    // FUN_END_TIME("GLUtils::CreateProgram")
    LOGCATE("GLUtils::CreateProgram program = %d", program);
	return program;
}

void testFSR(snn::GpuContext* context, const std::string& framesDir, float scaleFactor) {
    printf("%d\n", __LINE__);
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        printf("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/fsr/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        printf("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }   
    closedir(dir);
    if (frames.size() < 2) {
        return;
    }

    float resMultiplier = 2.0f;
    float rcasAtt = 0.25f;

    struct FSRConstants fsrData = {};

    for (size_t i = 0; i < frames.size(); ++i) {
        const std::string& frameName = frames[i % frames.size()].origName;
        uint32_t inputTexture = 0;

        bool ret = LoadTextureFromFile( (framesDir + "/" + frameName).c_str(), &inputTexture, &fsrData.input.width, &fsrData.input.height);

        printf("SR on %s, idx:%d\n", frameName.c_str(), inputTexture);

        fsrData.output = { (uint32_t)(fsrData.input.width * resMultiplier), (uint32_t)(fsrData.input.height * resMultiplier) };

        prepareFSR(&fsrData, rcasAtt);

        const std::string baseDir = "src/";

        uint32_t fsrProgramEASU = createFSRComputeProgramEAUS(baseDir);
        uint32_t fsrProgramRCAS = createFSRComputeProgramRCAS(baseDir);
        uint32_t bilinearProgram = createBilinearComputeProgram(baseDir);

        uint32_t outputImage = createOutputImage(fsrData);

        // upload the FSR constants, this contains the EASU and RCAS constants in a single uniform
        // TODO destroy the buffer
        unsigned int fsrData_vbo;
        {
            glGenBuffers(1, &fsrData_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, fsrData_vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(fsrData), &fsrData, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        printf("SR input:%d, output:%d\n", inputTexture, outputImage);

        runFSR(fsrData, fsrProgramEASU, fsrProgramRCAS, fsrData_vbo, inputTexture, outputImage);
        printf("SR done\n");

        std::array<uint32_t, 4> dims = {fsrData.output.width, fsrData.output.height, 1, 1};
        snn::ColorFormat cf = snn::ColorFormat::RGBA32F;
        snn::ImageTextureGL texExtr(dims, cf);
        texExtr.upload();
       #if 1 
        gl::TextureObject* glTex1 = texExtr.texture();
 
        glTex1->attach(GL_TEXTURE_2D, outputImage);
        texExtr.download();
        std::string extrName = frameName;
        texExtr.getRawImage().saveToPNG(outDir + extrName);
        texExtr.getRawImage().saveToBIN(outDir + extrName + ".bin");
        #endif
        printf("Save SR done\n");
    }
}

GLuint m_ImageTextureId;
GLuint m_FboTextureId;
GLuint m_FboId;
GLuint m_VaoIds[2];
GLuint m_VboIds[4];
GLint m_SamplerLoc;
GLuint m_FboProgramObj;
GLuint m_FboVertexShader;
GLuint m_FboFragmentShader;
GLint m_FboSamplerLoc;
GLint m_FboSamplerLoc1;
GLint m_MotionSamplerLoc;
GLint m_OpticalFlowHSScale;
GLint m_OpticalFlowHSOffset;
GLint m_OpticalFlowHSLamda;
static bool initialized = 0;

static void runGSR(struct FSRConstants fsrData, uint32_t inputImage, uint32_t outputImage) {
    //顶点坐标
	GLfloat vVertices[] = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
	};


	//正常纹理坐标
	GLfloat vTexCoors[] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
    };

	//fbo 纹理坐标与正常纹理方向不同，原点位于左下角
	GLfloat vFboTexCoors[] = {
			0.0f, 0.0f,
			1.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
	};

	GLushort indices[] = { 0, 1, 2, 1, 3, 2 };

	char vShaderStr[] =
			"#version 320 es                            \n"
			"layout(location = 0) in vec4 a_position;   \n"
			"layout(location = 1) in vec2 a_texCoord;   \n"
			"out vec2 v_texCoord;                       \n"
			"void main()                                \n"
			"{                                          \n"
			"   gl_Position = a_position;               \n"
			"   v_texCoord = a_texCoord;                \n"
			"}                                          \n";

#if 1
    const char *samplerName = "ps0";
    char fFboShaderStr[] = 
    R"(#version 320 es

    //============================================================================================================
    //
    //
    //                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
    //                              SPDX-License-Identifier: BSD-3-Clause
    //
    //============================================================================================================

    precision mediump float;
    precision highp int;

    ////////////////////////
    // USER CONFIGURATION //
    ////////////////////////

    /*
    * Operation modes:
    * RGBA -> 1
    * RGBY -> 3
    * LERP -> 4
    */
    #define OperationMode 1

    #define EdgeThreshold 8.0/255.0

    #define EdgeSharpness 2.0

    // #define UseUniformBlock

    ////////////////////////
    ////////////////////////
    ////////////////////////
    in highp vec2 v_texCoord;
    #if defined(UseUniformBlock)
    layout (set=0, binding = 0) uniform UniformBlock
    {
        highp vec4 ViewportInfo[1];
    };
    layout(set = 0, binding = 1) uniform mediump sampler2D ps0;
    #else
    //uniform highp vec4 ViewportInfo[1];
    uniform mediump sampler2D ps0;
    #endif

    //layout(location=0) in highp vec4 in_TEXCOORD0;
    layout(location=0) out vec4 out_Target0;

    float fastLanczos2(float x)
    {
        float wA = x-4.0;
        float wB = x*wA-wA;
        wA *= wA;
        return wB*wA;
    }
    vec2 weightY(float dx, float dy,float c, float std)
    {
        float x = ((dx*dx)+(dy* dy))* 0.55 + clamp(abs(c)*std, 0.0, 1.0);
        float w = fastLanczos2(x);
        return vec2(w, w * c);	
    }

    void main()
    {
        int mode = OperationMode;
        float edgeThreshold = EdgeThreshold;
        float edgeSharpness = EdgeSharpness;
        // highp vec2 myTexCoord = v_texCoord;
        highp vec2 myTexCoord = gl_FragCoord.xy;
        vec4 color;
        if(mode == 1)
            color.xyz = textureLod(ps0,v_texCoord.xy,0.0).xyz;
        else
            color.xyzw = textureLod(ps0,v_texCoord.xy,0.0).xyzw;

        highp float xCenter;
        xCenter = abs(v_texCoord.x+-0.5);
        highp float yCenter;
        yCenter = abs(v_texCoord.y+-0.5);

        //todo: config the SR region based on needs
        //if ( mode!=4 && xCenter*xCenter+yCenter*yCenter<=0.4 * 0.4)
        if ( mode!=4)
        {
            highp ivec2 inputImgSize = textureSize(ps0, 0).xy;
            highp vec4 myViewportInfo = vec4(0, 0, inputImgSize.x, inputImgSize.y);
            highp vec2 imgCoord = ((v_texCoord.xy*myViewportInfo.zw)+vec2(-0.5,0.5));
            highp vec2 imgCoordPixel = floor(imgCoord);
            highp vec2 coord = (imgCoordPixel*myViewportInfo.xy);
            vec2 pl = (imgCoord+(-imgCoordPixel));
            vec4  left = textureGather(ps0,coord, mode);

            float edgeVote = abs(left.z - left.y) + abs(color[mode] - left.y)  + abs(color[mode] - left.z) ;
            if(edgeVote > edgeThreshold)
            {
                coord.x += myViewportInfo.x;

                vec4 right = textureGather(ps0,coord + vec2(myViewportInfo.x, 0.0), mode);
                vec4 upDown;
                upDown.xy = textureGather(ps0,coord + vec2(0.0, -myViewportInfo.y),mode).wz;
                upDown.zw  = textureGather(ps0,coord+ vec2(0.0, myViewportInfo.y), mode).yx;

                float mean = (left.y+left.z+right.x+right.w)*0.25;
                left = left - vec4(mean);
                right = right - vec4(mean);
                upDown = upDown - vec4(mean);
                color.w =color[mode] - mean;

                float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));				
                float std = 2.181818/sum;
                
                vec2 aWY = weightY(pl.x, pl.y+1.0, upDown.x,std);				
                aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y,std);
                aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z,std);
                aWY += weightY(pl.x, pl.y-2.0, upDown.w,std);			
                aWY += weightY(pl.x+1.0, pl.y-1.0, left.x,std);
                aWY += weightY(pl.x, pl.y-1.0, left.y,std);
                aWY += weightY(pl.x, pl.y, left.z,std);
                aWY += weightY(pl.x+1.0, pl.y, left.w,std);
                aWY += weightY(pl.x-1.0, pl.y-1.0, right.x,std);
                aWY += weightY(pl.x-2.0, pl.y-1.0, right.y,std);
                aWY += weightY(pl.x-2.0, pl.y, right.z,std);
                aWY += weightY(pl.x-1.0, pl.y, right.w,std);

                float finalY = aWY.y/aWY.x;

                float maxY = max(max(left.y,left.z),max(right.x,right.w));
                float minY = min(min(left.y,left.z),min(right.x,right.w));
                finalY = clamp(edgeSharpness*finalY, minY, maxY);
                        
                float deltaY = finalY -color.w;	
                
                //smooth high contrast input
                deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

                color.x = clamp((color.x+deltaY),0.0,1.0);
                color.y = clamp((color.y+deltaY),0.0,1.0);
                color.z = clamp((color.z+deltaY),0.0,1.0);
            }
        }
        // color.xyz = textureLod(ps0,v_texCoord.xy,0.0).xyz;
        color.w = 1.0;  //assume alpha channel is not used
        out_Target0.xyzw = color;
    }
    )";
#endif
    if (!initialized) {
        // 编译链接用于离屏渲染的着色器程序
        m_FboProgramObj = CreateProgram(vShaderStr, fFboShaderStr, m_FboVertexShader, m_FboFragmentShader);

        if (m_FboProgramObj == GL_NONE)
        {
            LOGCATE("CopyTextureExample::Init m_ProgramObj == GL_NONE");
            return;
        }
        m_FboSamplerLoc = glGetUniformLocation(m_FboProgramObj, samplerName);

        // 生成 VBO ，加载顶点数据和索引数据
        // Generate VBO Ids and load the VBOs with data
        glGenBuffers(4, m_VboIds);
        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vVertices), vVertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vTexCoors), vTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vFboTexCoors), vFboTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // 生成 2 个 VAO，一个用于普通渲染，另一个用于离屏渲染
        // Generate VAO Ids
        glGenVertexArrays(2, m_VaoIds);

        // 初始化用于离屏渲染的 VAO
        // FBO off screen rendering VAO
        glBindVertexArray(m_VaoIds[1]);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glEnableVertexAttribArray(VERTEX_POS_INDX);
        glVertexAttribPointer(VERTEX_POS_INDX, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glEnableVertexAttribArray(TEXTURE_POS_INDX);
        glVertexAttribPointer(TEXTURE_POS_INDX, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        GO_CHECK_GL_ERROR();
        glBindVertexArray(GL_NONE);

        // 创建并初始化 FBO 纹理
        // glGenTextures(1, &m_FboTextureId);
        // glBindTexture(GL_TEXTURE_2D, m_FboTextureId);
        // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // glBindTexture(GL_TEXTURE_2D, GL_NONE);

        // 创建并初始化 FBO
        glGenFramebuffers(1, &m_FboId);
        initialized = 1;
    }

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glBindTexture(GL_TEXTURE_2D, outputImage);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputImage, 0);
	// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fsrData.output.width, fsrData.output.width, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
		LOGCATE("CopyTextureExample::CreateFrameBufferObj glCheckFramebufferStatus status != GL_FRAMEBUFFER_COMPLETE");
	}
	glBindTexture(GL_TEXTURE_2D, GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

	// 离屏渲染
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glViewport(0, 0, fsrData.output.width, fsrData.output.height);

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glUseProgram(m_FboProgramObj);
	glBindVertexArray(m_VaoIds[1]);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, inputImage);
	glUniform1i(m_FboSamplerLoc, 0);
	GO_CHECK_GL_ERROR();
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
	GO_CHECK_GL_ERROR();
	glBindVertexArray(0);

	//拷贝纹理
	// glBindTexture(GL_TEXTURE_2D, outputImage);
	// //glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_RenderImage.width, m_RenderImage.height);
    // glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, fsrData.output.width, fsrData.output.width, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    printf("%s:%d\n",__FUNCTION__, __LINE__);
}

void testGSR(snn::GpuContext* context, const std::string& framesDir, float scaleFactor) {
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        printf("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/gsr/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        printf("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }   
    closedir(dir);
    if (frames.size() < 2) {
        return;
    }
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );
    std::string frameNamePrev;
    std::shared_ptr<snn::ImageTexture> texPrev;
    std::array<uint32_t, 4> dims;
    snn::ColorFormat cf;

#ifdef PROFILING
    std::vector<std::unique_ptr<DeviceTimer>> gpuRunTimes(frames.size());
#endif
    for (size_t i = 0; i < frames.size(); ++i) {
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        printf("%s\n", tex->getTextureInfo2().c_str());
        tex->upload();

        dims = tex->getDims();
        cf = tex->format();
        SNN_ASSERT(tex->getDims() == dims);
        SNN_ASSERT(tex->format() == cf);

        struct FSRConstants fsrData = {};
        fsrData.output = { (uint32_t)(dims[0] * scaleFactor), (uint32_t)(dims[1]* scaleFactor)};
        std::array<uint32_t, 4> upDims = {(uint32_t)(dims[0] * scaleFactor), (uint32_t)(dims[1]* scaleFactor), dims[2], dims[3]};

        snn::ImageTextureGL& texGl1 = snn::ImageTextureGL::cast(*tex);
        SNN_ASSERT(texGl1.isValid());
        gl::TextureObject* glTex1 = texGl1.texture();
        SNN_ASSERT(glTex1);
        GLuint glTexId1 = glTex1->id();
        snn::ImageTextureGL texExtr(upDims, snn::ColorFormat::RGBA32F);
        texExtr.upload();
        gl::TextureObject* glTexExtr = texExtr.texture();
        SNN_ASSERT(glTexExtr);
        GLuint glTexIdExtr = glTexExtr->id();
        printf("GSR %d %f to %d\n", glTexId1, scaleFactor, glTexIdExtr);
#ifdef PROFILING
        gpuRunTimes[i].reset(new gl::GpuTimeElapsedQuery("frame_interpolation"));
        gpuRunTimes[i]->start();
#endif
        runGSR(fsrData, glTexId1, glTexIdExtr);
#ifdef PROFILING
        gpuRunTimes[i]->stop();
#endif
        texExtr.download();
        std::string extrName = frameName;
        texExtr.getRawImage().saveToPNG(outDir + extrName);
        texExtr.getRawImage().saveToBIN(outDir + extrName + ".bin");
    }
#ifdef PROFILING
    snn::Averager<float> gpuStat;
    for (size_t i = 0; i < gpuRunTimes.size(); ++i) {
        gpuRunTimes[i]->getTime();
        gpuStat.update(gpuRunTimes[i]->duration() / 1000.0);
    }
    printf("%s min: %f, max: %f, ave: %f mcs\n", "GSR", gpuStat.low, gpuStat.high, gpuStat.average);
#endif
}

static void runOpticalFlowHS(uint32_t inputImage0, uint32_t inputImage1, uint32_t outputImage, uint32_t width, uint32_t height) {
    //顶点坐标
	GLfloat vVertices[] = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
	};


	//正常纹理坐标
	GLfloat vTexCoors[] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
    };

	//fbo 纹理坐标与正常纹理方向不同，原点位于左下角
	GLfloat vFboTexCoors[] = {
			0.0f, 0.0f,
			1.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
	};

	GLushort indices[] = { 0, 1, 2, 1, 3, 2 };

	char vShaderStr[] =
			"#version 320 es                            \n"
			"layout(location = 0) in vec4 a_position;   \n"
			"layout(location = 1) in vec2 a_texCoord;   \n"
			"out vec2 texCoord;                       \n"
			"void main()                                \n"
			"{                                          \n"
			"   gl_Position = a_position;               \n"
			"   texCoord = a_texCoord;                \n"
			"}                                          \n";

#if 1
    const char *samplerName = "tex0";
    const char *samplerName1 = "tex1";
    char fFboShaderStr[] = 
    R"(#version 320 es
    precision highp float;
    uniform sampler2D tex0;  
	uniform sampler2D tex1;  

	uniform vec2 scale;  
	uniform vec2 offset;  
	uniform float lambda;   
	in vec2 texCoord;
    layout(location=0) out vec4 gl_FragColor;

	vec4 getColorCoded(float x, float y,vec2 scale) {
		vec2 xout = vec2(max(x,0.),abs(min(x,0.)))*scale.x;
		vec2 yout = vec2(max(y,0.),abs(min(y,0.)))*scale.y;
		float dirY = 1.0f;
		if (yout.x > yout.y)  dirY=0.90;
		//return vec4(xout.xy,max(yout.x,yout.y),dirY);
        return vec4(x,y,0.0f,dirY);
	}


	vec4 getGrayScale(vec4 col) {
		float gray = dot(vec3(col.x, col.y, col.z), vec3(0.3, 0.59, 0.11));
		return vec4(gray,gray,gray,1);
	}
	vec4 texture2DRectGray(sampler2D tex, vec2 coord) {
		return getGrayScale(texture(tex, coord));
	}

	void main()  
	{     
		vec4 a = texture2DRectGray(tex0, texCoord);
		vec4 b = texture2DRectGray(tex1, texCoord);
		vec2 x1 = vec2(offset.x,0.);
		vec2 y1 = vec2(0.,offset.y);

		//get the difference
		vec4 curdif = b-a;

		//calculate the gradient
		//for X________________
		vec4 gradx = texture2DRectGray(tex1, texCoord+x1)-texture2DRectGray(tex1, texCoord-x1);
		gradx += texture2DRectGray(tex0, texCoord+x1)-texture2DRectGray(tex0, texCoord-x1);


		//for Y________________
		vec4 grady = texture2DRectGray(tex1, texCoord+y1)-texture2DRectGray(tex1, texCoord-y1);
		grady += texture2DRectGray(tex0, texCoord+y1)-texture2DRectGray(tex0, texCoord-y1);

		vec4 gradmag = sqrt((gradx*gradx)+(grady*grady)+vec4(lambda));

		vec4 vx = curdif*(gradx/gradmag);
		vec4 vy = curdif*(grady/gradmag);

		gl_FragColor = getColorCoded(vx.r,vy.r,scale);   
	}
    )";
#endif
    if (!initialized) {
        // 编译链接用于离屏渲染的着色器程序
        m_FboProgramObj = CreateProgram(vShaderStr, fFboShaderStr, m_FboVertexShader, m_FboFragmentShader);

        if (m_FboProgramObj == GL_NONE)
        {
            LOGCATE("CopyTextureExample::Init m_ProgramObj == GL_NONE");
            return;
        }
        m_FboSamplerLoc = glGetUniformLocation(m_FboProgramObj, samplerName);
        m_FboSamplerLoc1 = glGetUniformLocation(m_FboProgramObj, samplerName1);

        m_OpticalFlowHSScale = glGetUniformLocation(m_FboProgramObj, "scale");
        m_OpticalFlowHSOffset = glGetUniformLocation(m_FboProgramObj, "offset");
        m_OpticalFlowHSLamda = glGetUniformLocation(m_FboProgramObj, "lamda");

        // 生成 VBO ，加载顶点数据和索引数据
        // Generate VBO Ids and load the VBOs with data
        glGenBuffers(4, m_VboIds);
        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vVertices), vVertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vTexCoors), vTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vFboTexCoors), vFboTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // 生成 2 个 VAO，一个用于普通渲染，另一个用于离屏渲染
        // Generate VAO Ids
        glGenVertexArrays(2, m_VaoIds);

        // 初始化用于离屏渲染的 VAO
        // FBO off screen rendering VAO
        glBindVertexArray(m_VaoIds[1]);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glEnableVertexAttribArray(VERTEX_POS_INDX);
        glVertexAttribPointer(VERTEX_POS_INDX, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glEnableVertexAttribArray(TEXTURE_POS_INDX);
        glVertexAttribPointer(TEXTURE_POS_INDX, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        GO_CHECK_GL_ERROR();
        glBindVertexArray(GL_NONE);

        // 创建并初始化 FBO
        glGenFramebuffers(1, &m_FboId);
        initialized = 1;
    }

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glBindTexture(GL_TEXTURE_2D, outputImage);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputImage, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
		LOGCATE("CopyTextureExample::CreateFrameBufferObj glCheckFramebufferStatus status != GL_FRAMEBUFFER_COMPLETE");
	}
	glBindTexture(GL_TEXTURE_2D, GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

	// 离屏渲染
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glViewport(0, 0, width, height);

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glUseProgram(m_FboProgramObj);
	glBindVertexArray(m_VaoIds[1]);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, inputImage0);
	glUniform1i(m_FboSamplerLoc, 0);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, inputImage1);
	glUniform1i(m_FboSamplerLoc1, 1);
	GO_CHECK_GL_ERROR();


	glUniform2f(m_OpticalFlowHSScale, 1.0f, 1.0f);
    glUniform2f(m_OpticalFlowHSOffset, 1.0f, 1.0f);
    glUniform1f(m_OpticalFlowHSLamda, 0.1f);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
	GO_CHECK_GL_ERROR();
	glBindVertexArray(0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    printf("%s:%d\n",__FUNCTION__, __LINE__);
}

void testOpticalFlowHS(snn::GpuContext* context, const std::string& framesDir, float scaleFactor) {
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        SNN_RIP("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/ofhs/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        SNN_LOGI("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }   
    closedir(dir);
    if (frames.size() < 2) {
        return;
    }
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );
    std::string frameNamePrev;
    std::shared_ptr<snn::ImageTexture> texPrev;
    std::array<uint32_t, 4> dims;
    snn::ColorFormat cf;

#ifdef PROFILING
    std::vector<std::unique_ptr<DeviceTimer>> gpuRunTimes(frames.size());
#endif
    for (size_t i = 0; i <= frames.size(); ++i) {
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        printf("%s\n", tex->getTextureInfo2().c_str());
        tex->upload();
        if (i > 0) {
            SNN_ASSERT(tex->getDims() == dims);
            SNN_ASSERT(tex->format() == cf);
            snn::ImageTextureGL& texGl0 = snn::ImageTextureGL::cast(*texPrev);
            snn::ImageTextureGL& texGl1 = snn::ImageTextureGL::cast(*tex);
            SNN_ASSERT(texGl0.isValid());
            SNN_ASSERT(texGl1.isValid());
            gl::TextureObject* glTex0 = texGl0.texture();
            SNN_ASSERT(glTex0);
            gl::TextureObject* glTex1 = texGl1.texture();
            SNN_ASSERT(glTex1);
            GLuint glTexId0 = glTex0->id();
            GLuint glTexId1 = glTex1->id();
            snn::ImageTextureGL texExtr(dims, cf);
            texExtr.upload();
            gl::TextureObject* glTexExtr = texExtr.texture();
            SNN_ASSERT(glTexExtr);
            GLuint glTexIdExtr = glTexExtr->id();
            printf("Extrapolating %d with %d for %d factor %f\n", glTexId0, glTexId1, glTexIdExtr, scaleFactor);
#ifdef PROFILING
            gpuRunTimes[i - 1].reset(new gl::GpuTimeElapsedQuery("frame_interpolation"));
            gpuRunTimes[i - 1]->start();
#endif
            runOpticalFlowHS(glTexId0, glTexId1, glTexIdExtr, dims[0], dims[1]);
#ifdef PROFILING
            gpuRunTimes[i - 1]->stop();
#endif
            texExtr.download();
            std::string extrName = extrapolName(frameNamePrev, frameName, scaleFactor);
            printf("Extrapolated %s and %s with %f saved in %s\n", frameNamePrev.c_str(), frameName.c_str(), scaleFactor, extrName.c_str());
            texExtr.getRawImage().saveToPNG(outDir + extrName);
            texExtr.getRawImage().saveToBIN(outDir + extrName + ".bin");
        } else {
            dims = tex->getDims();
            cf = tex->format();
        }
        texPrev =  tex;
        frameNamePrev = frameName;
    }
#ifdef PROFILING
    snn::Averager<float> gpuStat;
    for (size_t i = 0; i < gpuRunTimes.size(); ++i) {
        gpuRunTimes[i]->getTime();
        gpuStat.update(gpuRunTimes[i]->duration() / 1000.0);
    }
    printf("%s min: %f, max: %f, ave: %f mcs\n", F_EXTRAPOLATION_FUN_NAME, gpuStat.low, gpuStat.high, gpuStat.average);
#endif
}

static void runWarp(uint32_t inputImage0, uint32_t inputImage1, uint32_t mvImage, uint32_t outputImage, uint32_t width, uint32_t height) {
    //顶点坐标
	GLfloat vVertices[] = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
	};


	//正常纹理坐标
	GLfloat vTexCoors[] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
    };

	//fbo 纹理坐标与正常纹理方向不同，原点位于左下角
	GLfloat vFboTexCoors[] = {
			0.0f, 0.0f,
			1.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,
	};

	GLushort indices[] = { 0, 1, 2, 1, 3, 2 };

	char vShaderStr[] =
			"#version 320 es                            \n"
			"layout(location = 0) in vec4 a_position;   \n"
			"layout(location = 1) in vec2 a_texCoord;   \n"
			"out vec2 texCoord;                       \n"
			"void main()                                \n"
			"{                                          \n"
			"   gl_Position = a_position;               \n"
			"   texCoord = a_texCoord;                \n"
			"}                                          \n";

#if 1
    const char *samplerName = "tex0";
    const char *samplerName1 = "tex1";
    const char *motionName = "mv0";
    char fFboShaderStr[] = 
    R"(#version 320 es
    precision highp float;
    uniform sampler2D tex0;  
	uniform sampler2D tex1;  
	uniform sampler2D mv0;  

	uniform vec2 scale;  
	uniform vec2 offset;  
	uniform float lambda;   
	in vec2 texCoord;
    layout(location=0) out vec4 gl_FragColor;

	vec4 getColorCoded(float x, float y,vec2 scale) {
		vec2 xout = vec2(max(x,0.),abs(min(x,0.)))*scale.x;
		vec2 yout = vec2(max(y,0.),abs(min(y,0.)))*scale.y;
		float dirY = 1.0f;
		if (yout.x > yout.y)  dirY=0.90;
		//return vec4(xout.xy,max(yout.x,yout.y),dirY);
        return vec4(x,y,0.0f,dirY);
	}

	vec4 getGrayScale(vec4 col) {
		float gray = dot(vec3(col.x, col.y, col.z), vec3(0.3, 0.59, 0.11));
		return vec4(gray,gray,gray,1);
	}
	vec4 texture2DRectGray(sampler2D tex, vec2 coord) {
		return getGrayScale(texture(tex, coord));
	}

	void main()  
	{     
        vec4 m = texture(mv0, texCoord);
        // m.x = clamp(m.x, -5.0, 5.0);
        // m.y = clamp(m.y, -5.0, 5.0);
        vec2 newPos = vec2(gl_FragCoord.x - m.x/2.0*8.0, gl_FragCoord.y - m.y/2.0*8.0);
        ivec2 size = textureSize(tex0, 0);
        vec2 currLoc = vec2(clamp(newPos.x, 0.0f, float(size.x-1)), clamp(newPos.y, 0.0f, float(size.y-1)));
        ivec2 outLoc = ivec2(currLoc.x, currLoc.y);
		// ivec2 outLoc = ivec2(gl_FragCoord.xy);
        vec4 a = texelFetch(tex0, outLoc, 0);
		gl_FragColor = vec4(a.xyz, 1.0f);
        //vec4 a = texelFetch(tex0, ivec2(gl_FragCoord.xy), 0);
        gl_FragColor = vec4(m.xy, 0, 255);
        //gl_FragColor = vec4(vec2(128,128), 0, 1);
	}
    )";
#endif
    if (!initialized) {
        // 编译链接用于离屏渲染的着色器程序
        m_FboProgramObj = CreateProgram(vShaderStr, fFboShaderStr, m_FboVertexShader, m_FboFragmentShader);

        if (m_FboProgramObj == GL_NONE)
        {
            LOGCATE("CopyTextureExample::Init m_ProgramObj == GL_NONE");
            return;
        }
        m_FboSamplerLoc = glGetUniformLocation(m_FboProgramObj, samplerName);
        m_FboSamplerLoc1 = glGetUniformLocation(m_FboProgramObj, samplerName1);
        m_MotionSamplerLoc = glGetUniformLocation(m_FboProgramObj, motionName);

        m_OpticalFlowHSScale = glGetUniformLocation(m_FboProgramObj, "scale");
        m_OpticalFlowHSOffset = glGetUniformLocation(m_FboProgramObj, "offset");
        m_OpticalFlowHSLamda = glGetUniformLocation(m_FboProgramObj, "lamda");

        // 生成 VBO ，加载顶点数据和索引数据
        // Generate VBO Ids and load the VBOs with data
        glGenBuffers(4, m_VboIds);
        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vVertices), vVertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vTexCoors), vTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vFboTexCoors), vFboTexCoors, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // 生成 2 个 VAO，一个用于普通渲染，另一个用于离屏渲染
        // Generate VAO Ids
        glGenVertexArrays(2, m_VaoIds);

        // 初始化用于离屏渲染的 VAO
        // FBO off screen rendering VAO
        glBindVertexArray(m_VaoIds[1]);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[0]);
        glEnableVertexAttribArray(VERTEX_POS_INDX);
        glVertexAttribPointer(VERTEX_POS_INDX, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ARRAY_BUFFER, m_VboIds[2]);
        glEnableVertexAttribArray(TEXTURE_POS_INDX);
        glVertexAttribPointer(TEXTURE_POS_INDX, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, GL_NONE);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_VboIds[3]);
        GO_CHECK_GL_ERROR();
        glBindVertexArray(GL_NONE);

        // 创建并初始化 FBO
        glGenFramebuffers(1, &m_FboId);
        initialized = 1;
    }

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glBindTexture(GL_TEXTURE_2D, outputImage);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputImage, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
		LOGCATE("CopyTextureExample::CreateFrameBufferObj glCheckFramebufferStatus status != GL_FRAMEBUFFER_COMPLETE");
	}
	glBindTexture(GL_TEXTURE_2D, GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

	// 离屏渲染
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);
	glViewport(0, 0, width, height);

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboId);
	glUseProgram(m_FboProgramObj);
	glBindVertexArray(m_VaoIds[1]);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, inputImage0);
	glUniform1i(m_FboSamplerLoc, 0);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, inputImage1);
	glUniform1i(m_FboSamplerLoc1, 1);
	glActiveTexture(GL_TEXTURE0 + 2);
	glBindTexture(GL_TEXTURE_2D, mvImage);
	glUniform1i(m_MotionSamplerLoc, 2);
	GO_CHECK_GL_ERROR();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glUniform2f(m_OpticalFlowHSScale, 1.0f, 1.0f);
    glUniform2f(m_OpticalFlowHSOffset, 1.0f, 1.0f);
    glUniform1f(m_OpticalFlowHSLamda, 0.1f);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
	GO_CHECK_GL_ERROR();
	glBindVertexArray(0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    printf("%s:%d\n",__FUNCTION__, __LINE__);
}

void testWarp(snn::GpuContext* context, const std::string& framesDir, float scaleFactor,  P_ESTIMATE_M estimateM) {
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        SNN_RIP("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/warp/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        SNN_LOGI("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }   
    closedir(dir);
    if (frames.size() < 2) {
        return;
    }
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );
    std::string frameNamePrev;
    std::shared_ptr<snn::ImageTexture> texPrev;
    std::array<uint32_t, 4> dims;
    snn::ColorFormat cf;

    GLint motionBlockSizeX;
    GLint motionBlockSizeY;
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &motionBlockSizeX));
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &motionBlockSizeY));
    printf("MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM - %d, MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM = %d\n", motionBlockSizeX, motionBlockSizeY);

#ifdef PROFILING
    std::vector<std::unique_ptr<DeviceTimer>> gpuRunTimes(frames.size());
#endif
    for (size_t i = 0; i <= frames.size(); ++i) {
        float maxM = 0.0f;
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        printf("To process: %s, %s\n", frameName.c_str(), tex->getTextureInfo2().c_str());
        tex->upload();
        if (i > 0) {
            SNN_ASSERT(tex->getDims() == dims);
            SNN_ASSERT(tex->format() == cf);
            snn::ImageTextureGL& texGl0 = snn::ImageTextureGL::cast(*texPrev);
            snn::ImageTextureGL& texGl1 = snn::ImageTextureGL::cast(*tex);
            SNN_ASSERT(texGl0.isValid());
            SNN_ASSERT(texGl1.isValid());
            gl::TextureObject* glTex0 = texGl0.texture();
            SNN_ASSERT(glTex0);
            gl::TextureObject* glTex1 = texGl1.texture();
            SNN_ASSERT(glTex1);
            GLuint glTexId0 = glTex0->id();
            GLuint glTexId1 = glTex1->id();
            snn::ImageTextureGL texY0(dims, snn::ColorFormat::R8);
            snn::ImageTextureGL texY1(dims, snn::ColorFormat::R8);
            rgba2Y(texGl0, texY0);
            rgba2Y(texGl1, texY1);
            texY0.upload();
            texY1.upload();
            gl::TextureObject* glTexY0 = texY0.texture();
            gl::TextureObject* glTexY1 = texY1.texture();
            SNN_ASSERT(glTexY0);
            SNN_ASSERT(glTexY1);
            GLuint glTexYId0 = glTexY0->id();
            GLuint glTexYId1 = glTexY1->id();
            snn::ImageTextureGL texMotion({dims[0] / motionBlockSizeX, dims[1] / motionBlockSizeY, 1, 1}, snn::ColorFormat::RGBA16F);
            texMotion.upload();
            gl::TextureObject* glTexMotion = texMotion.texture();
            SNN_ASSERT(glTexMotion);
            GLuint glTexIdMotion = glTexMotion->id();
#ifdef PROFILING
            gpuRunTimes[i - 1].reset(new gl::GpuTimeElapsedQuery("motion_estimation"));
            gpuRunTimes[i - 1]->start();
#endif
            SNN_LOGD("Calculating motion estimate %d with %d to %d", glTexId0, glTexId1, glTexIdMotion);
            GLCHK(estimateM(glTexYId0, glTexYId1, glTexIdMotion));
            SNN_LOGD("Calculated motion estimate");
#ifdef PROFILING
            gpuRunTimes[i - 1]->stop();
#endif      
            #if 1
            texMotion.download();
            texMotion.convertFormat(snn::ColorFormat::RGBA32F);
            snn::ImageTextureGL texMotionR({dims[0] / motionBlockSizeX, dims[1] / motionBlockSizeY, 1, 1}, snn::ColorFormat::R32F);
            rgba2R(texMotion, texMotionR, 1.0f /** 255.0f*/, &maxM);
            std::string mEstName = motionEstName(frameNamePrev, frameName);
            SNN_LOGI("Motion estimation %s and %s saved in %s\n", frameNamePrev.c_str(), frameName.c_str(), mEstName.c_str());
            SNN_LOGI("Maximum absolute value of motion vector is %f", maxM);
            texMotionR.getRawImage().saveToPNG(outDir + mEstName);
            texMotion.getRawImage().saveToBIN(outDir + mEstName + ".bin");
            #endif

            snn::ImageTextureGL texExtr(dims, snn::ColorFormat::RGBA32F);
            texExtr.upload();
            gl::TextureObject* glTexExtr = texExtr.texture();
            SNN_ASSERT(glTexExtr);
            GLuint glTexIdExtr = glTexExtr->id();
            printf("Extrapolating %d with %d for %d factor %f\n", glTexId0, glTexId1, glTexIdExtr, scaleFactor);
            texMotion.upload();
#ifdef PROFILING
            gpuRunTimes[i - 1].reset(new gl::GpuTimeElapsedQuery("frame_interpolation"));
            gpuRunTimes[i - 1]->start();
#endif
            runWarp(glTexId0, glTexId1, texMotion.texture()->id(), glTexIdExtr, dims[0], dims[1]);
#ifdef PROFILING
            gpuRunTimes[i - 1]->stop();
#endif
            texExtr.download();
            std::string extrName = extrapolName(frameNamePrev, frameName, scaleFactor);
            printf("Extrapolated %s and %s with %f saved in %s\n", frameNamePrev.c_str(), frameName.c_str(), scaleFactor, extrName.c_str());
            texExtr.getRawImage().saveToPNG(outDir + extrName);
            texExtr.getRawImage().saveToBIN(outDir + extrName + ".bin");

        } else {
            dims = tex->getDims();
            cf = tex->format();
        }
        texPrev =  tex;
        frameNamePrev = frameName;
    }
#ifdef PROFILING
    snn::Averager<float> gpuStat;
    for (size_t i = 0; i < gpuRunTimes.size(); ++i) {
        gpuRunTimes[i]->getTime();
        gpuStat.update(gpuRunTimes[i]->duration() / 1000.0);
    }
    printf("%s min: %f, max: %f, ave: %f mcs\n", F_EXTRAPOLATION_FUN_NAME, gpuStat.low, gpuStat.high, gpuStat.average);
#endif
}

#ifdef SUPPORT_GL
void extrapolateFrames(snn::GpuContext* context, const std::string& framesDir, float scaleFactor, P_EXTRAPOLATE_TEX extrapolateTex) {
    SNN_ASSERT(extrapolateTex);
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        SNN_RIP("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/extrapolated/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        SNN_LOGI("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }   
    closedir(dir);
    if (frames.size() < 2) {
        return;
    }
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );
    std::string frameNamePrev;
    std::shared_ptr<snn::ImageTexture> texPrev;
    std::array<uint32_t, 4> dims;
    snn::ColorFormat cf;

#ifdef PROFILING
    std::vector<std::unique_ptr<DeviceTimer>> gpuRunTimes(frames.size());
#endif
    for (size_t i = 0; i <= frames.size(); ++i) {
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        SNN_LOGI("%s", tex->getTextureInfo2().c_str());
        tex->upload();
        if (i > 0) {
            SNN_ASSERT(tex->getDims() == dims);
            SNN_ASSERT(tex->format() == cf);
            snn::ImageTextureGL& texGl0 = snn::ImageTextureGL::cast(*texPrev);
            snn::ImageTextureGL& texGl1 = snn::ImageTextureGL::cast(*tex);
            SNN_ASSERT(texGl0.isValid());
            SNN_ASSERT(texGl1.isValid());
            gl::TextureObject* glTex0 = texGl0.texture();
            SNN_ASSERT(glTex0);
            gl::TextureObject* glTex1 = texGl1.texture();
            SNN_ASSERT(glTex1);
            GLuint glTexId0 = glTex0->id();
            GLuint glTexId1 = glTex1->id();
            snn::ImageTextureGL texExtr(dims, cf);
            texExtr.upload();
            gl::TextureObject* glTexExtr = texExtr.texture();
            SNN_ASSERT(glTexExtr);
            GLuint glTexIdExtr = glTexExtr->id();
            SNN_LOGD("Extrapolating %d with %d for %f to %d", glTexId0, glTexId1, glTexIdExtr, scaleFactor);
#ifdef PROFILING
            gpuRunTimes[i - 1].reset(new gl::GpuTimeElapsedQuery("frame_interpolation"));
            gpuRunTimes[i - 1]->start();
#endif
            GLCHK(extrapolateTex(glTexId0, glTexId1, glTexIdExtr, scaleFactor));
#ifdef PROFILING
            gpuRunTimes[i - 1]->stop();
#endif
            texExtr.download();
            std::string extrName = extrapolName(frameNamePrev, frameName, scaleFactor);
            SNN_LOGI("Extrapolated %s and %s with %f saved in %s\n", frameNamePrev.c_str(), frameName.c_str(), scaleFactor, extrName.c_str());
            texExtr.getRawImage().saveToPNG(outDir + extrName);

#if 0 // Testing that we indeed extrapolating different images
            if (i == 2) {
                texGl0.getRawImage().saveToPNG(outDir + frameNamePrev);
                texGl1.getRawImage().saveToPNG(outDir + frameName);
            }
#endif
        } else {
            dims = tex->getDims();
            cf = tex->format();
        }
        texPrev =  tex;
        frameNamePrev = frameName;
    }
#ifdef PROFILING
    snn::Averager<float> gpuStat;
    for (size_t i = 0; i < gpuRunTimes.size(); ++i) {
        gpuRunTimes[i]->getTime();
        gpuStat.update(gpuRunTimes[i]->duration() / 1000.0);
    }
    SNN_LOGI("%s min: %f, max: %f, ave: %f mcs", F_EXTRAPOLATION_FUN_NAME, gpuStat.low, gpuStat.high, gpuStat.average);
#endif
}

void estimateMotion(snn::GpuContext* context, const std::string& framesDir, P_ESTIMATE_M estimateM) {
    SNN_ASSERT(estimateM);
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        SNN_RIP("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/motion/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        SNN_LOGI("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );
    std::string frameNamePrev;
    std::shared_ptr<snn::ImageTexture> texPrev;
    std::array<uint32_t, 4> dims;
    snn::ColorFormat cf;

    GLint motionBlockSizeX;
    GLint motionBlockSizeY;
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &motionBlockSizeX));
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &motionBlockSizeY));
    SNN_LOGI("MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM - %d, MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM = %d", motionBlockSizeX, motionBlockSizeY);
#ifdef PROFILING
    std::vector<std::unique_ptr<DeviceTimer>> gpuRunTimes(frames.size());
#endif
    for (size_t i = 0; i <= frames.size(); ++i) {
        float maxM = 0.0f;
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        tex->upload();
        if (i > 0) {
            SNN_ASSERT(tex->getDims() == dims);
            SNN_ASSERT(tex->format() == cf);
            snn::ImageTextureGL& texGl0 = snn::ImageTextureGL::cast(*texPrev);
            snn::ImageTextureGL& texGl1 = snn::ImageTextureGL::cast(*tex);
            SNN_ASSERT(texGl0.isValid());
            SNN_ASSERT(texGl1.isValid());
            gl::TextureObject* glTex0 = texGl0.texture();
            SNN_ASSERT(glTex0);
            gl::TextureObject* glTex1 = texGl1.texture();
            SNN_ASSERT(glTex1);
            GLuint glTexId0 = glTex0->id();
            GLuint glTexId1 = glTex1->id();
            snn::ImageTextureGL texY0(dims, snn::ColorFormat::R8);
            snn::ImageTextureGL texY1(dims, snn::ColorFormat::R8);
            rgba2Y(texGl0, texY0);
            rgba2Y(texGl1, texY1);
            texY0.upload();
            texY1.upload();
            gl::TextureObject* glTexY0 = texY0.texture();
            gl::TextureObject* glTexY1 = texY1.texture();
            SNN_ASSERT(glTexY0);
            SNN_ASSERT(glTexY1);
            GLuint glTexYId0 = glTexY0->id();
            GLuint glTexYId1 = glTexY1->id();
            snn::ImageTextureGL texMotion({dims[0] / motionBlockSizeX, dims[1] / motionBlockSizeY, 1, 1}, snn::ColorFormat::RGBA16F);
            texMotion.upload();
            gl::TextureObject* glTexMotion = texMotion.texture();
            SNN_ASSERT(glTexMotion);
            GLuint glTexIdMotion = glTexMotion->id();
#ifdef PROFILING
            gpuRunTimes[i - 1].reset(new gl::GpuTimeElapsedQuery("motion_estimation"));
            gpuRunTimes[i - 1]->start();
#endif
            SNN_LOGD("Calculating motion estimate %d with %d to %d", glTexId0, glTexId1, glTexIdMotion);
            GLCHK(estimateM(glTexYId0, glTexYId1, glTexIdMotion));
            SNN_LOGD("Calculated motion estimate");
#ifdef PROFILING
            gpuRunTimes[i - 1]->stop();
#endif      
            texMotion.download();
            texMotion.convertFormat(snn::ColorFormat::RGBA32F);
            snn::ImageTextureGL texMotionR({dims[0] / motionBlockSizeX, dims[1] / motionBlockSizeY, 1, 1}, snn::ColorFormat::R32F);
            rgba2R(texMotion, texMotionR, 1.0f /** 255.0f*/, &maxM);
            std::string mEstName = motionEstName(frameNamePrev, frameName);
            SNN_LOGI("Motion estimation %s and %s saved in %s\n", frameNamePrev.c_str(), frameName.c_str(), mEstName.c_str());
            SNN_LOGI("Maximum absolute value of motion vector is %f", maxM);
            texMotion.getRawImage().saveToPNG(outDir + mEstName);
            texMotion.getRawImage().saveToBIN(outDir + mEstName + ".bin");
#if 1
            FILE* fp;
            fp = fopen((outDir + frameNamePrev.c_str() + ".txt").c_str(), "w");
            texY0.prettyPrint(fp);
            fclose(fp);

            fp = fopen((outDir + frameName.c_str() + ".txt").c_str(), "w");
            texY1.prettyPrint(fp);
            fclose(fp);

            fp = fopen((outDir + mEstName + ".txt").c_str(), "w");
            texMotionR.prettyPrint(fp);
            fclose(fp);
#endif
        } else {
            dims = tex->getDims();
            cf = tex->format();
        }
        texPrev =  tex;
        frameNamePrev = frameName;
    }
#ifdef PROFILING
    snn::Averager<float> gpuStat;
    for (size_t i = 0; i < gpuRunTimes.size(); ++i) {
        gpuRunTimes[i]->getTime();
        gpuStat.update(gpuRunTimes[i]->duration() / 1000.0);
    }
    SNN_LOGI("%s min: %f, max: %f, ave: %f mcs", F_M_ESTIMATION_FUN_NAME, gpuStat.low, gpuStat.high, gpuStat.average);
#endif
}

void estimateMotionSynthetic(snn::GpuContext* context, const std::string& framesDir, P_ESTIMATE_M estimateM, int offset) {
    SNN_ASSERT(estimateM);
    DIR *dir;
    if ((dir = opendir(framesDir.c_str())) == NULL) {
        SNN_RIP("Directory '%s' does not exist!", framesDir.c_str());
    }
    const std::string outDir = framesDir + "/motion_synth/";
    struct dirent *entry;
    std::vector<FileNameToSort> frames;
    while ((entry = readdir(dir)) != NULL) {
        SNN_LOGI("%s\n", entry->d_name);
        if (entry->d_type == DT_REG) {
            frames.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    std::sort(frames.begin(), frames.end(), [](const auto& el1, const auto& el2) {return el1.paddedName < el2.paddedName;} );

    GLint motionBlockSizeX;
    GLint motionBlockSizeY;
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &motionBlockSizeX));
    GLCHK(glGetIntegerv(MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &motionBlockSizeY));
    SNN_LOGI("MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM - %d, MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM = %d", motionBlockSizeX, motionBlockSizeY);

    for (size_t i = 0; i <= frames.size(); ++i) {
        const std::string& frameName = frames[i % frames.size()].origName;
        std::shared_ptr<snn::ImageTexture> tex = snn::ImageTextureFactory::createImageTexture(context, framesDir + '/' + frameName);
        tex->upload();
        snn::ImageTextureGL& texGl0 = snn::ImageTextureGL::cast(*tex);
        auto dims = texGl0.getDims();
        snn::ImageTextureGL texGl1(dims, texGl0.format());
        offsetTex(texGl0, texGl1, offset);
        texGl1.upload();
        SNN_ASSERT(texGl0.isValid());
        SNN_ASSERT(texGl1.isValid());
        gl::TextureObject* glTex0 = texGl0.texture();
        SNN_ASSERT(glTex0);
        gl::TextureObject* glTex1 = texGl1.texture();
        SNN_ASSERT(glTex1);
        GLuint glTexId0 = glTex0->id();
        GLuint glTexId1 = glTex1->id();
        snn::ImageTextureGL texY0(dims, snn::ColorFormat::R8);
        snn::ImageTextureGL texY1(dims, snn::ColorFormat::R8);
        rgba2Y(texGl0, texY0);
        rgba2Y(texGl1, texY1);
        texY0.upload();
        texY1.upload();
        gl::TextureObject* glTexY0 = texY0.texture();
        gl::TextureObject* glTexY1 = texY1.texture();
        SNN_ASSERT(glTexY0);
        SNN_ASSERT(glTexY1);
        GLuint glTexYId0 = glTexY0->id();
        GLuint glTexYId1 = glTexY1->id();
        snn::ImageTextureGL texMotion({dims[0] / motionBlockSizeX, dims[1] / motionBlockSizeY, 1, 1}, snn::ColorFormat::RGBA16F);
        texMotion.upload();
        gl::TextureObject* glTexMotion = texMotion.texture();
        SNN_ASSERT(glTexMotion);
        GLuint glTexIdMotion = glTexMotion->id();
        SNN_LOGD("Calculating motion estimate %d with %d to %d", glTexId0, glTexId1, glTexIdMotion);
        GLCHK(estimateM(glTexYId0, glTexYId1, glTexIdMotion));
        SNN_LOGD("Calculated motion estimate");
        texMotion.download();
        texMotion.convertFormat(snn::ColorFormat::RGBA32F);
        snn::ImageTextureGL texMotionR({texMotion.getDims()[0], texMotion.getDims()[1], 1, 1}, snn::ColorFormat::R32F);
        rgba2R(texMotion, texMotionR);
        std::string mEstName = motionEstName(frameName, frameName + "_" + std::to_string(offset));
        SNN_LOGI("Motion estimation of %s with offset %d saved in %s\n", frameName.c_str(), offset, mEstName.c_str());
        texMotionR.getRawImage().saveToPNG(outDir + mEstName);
#if 0
        FILE* fp;
        fp = fopen((outDir + frameName.c_str() + ".txt").c_str(), "w");
        texY0.prettyPrint(fp);
        fclose(fp);

        fp = fopen((outDir + frameName.c_str() + "_" + std::to_string(offset) + ".txt").c_str(), "w");
        texY1.prettyPrint(fp);
        fclose(fp);
#endif
#if 1
        FILE* fp;
        fp = fopen((outDir + mEstName + ".txt").c_str(), "w");
        texMotionR.prettyPrint(fp);
        fclose(fp);
#endif
    }
}
#endif

int main(int argc, char **argv) {
    CHECK_PLATFORM_SUPPORT(false)

    std::string framesDir;
    float scaleFactor =  0.0f;
    int offset = 0;
    int test = 0;
    CLI::App app;
    app.add_option("frames_dir", framesDir, "Directory with frame files");
    app.add_option("--scale_factor", scaleFactor);
    app.add_option("--offset", offset);
    app.add_set("--test", test, {0, 1, 2, 3, 4, 5, 6}, "Test type. 0 = Frame extrapolation, 1 = Motion estimation, 2 = Synthetic motion estimation, 3 = GSR, 4 = Optical Flow, 5 = FSR, 6 = Warp");
    CLI11_PARSE(app, argc, argv);

#ifdef SUPPORT_GL
    snn::GpuContext* context = snn::createDefaultContext(false);

    P_EXTRAPOLATE_TEX extrapolateTex = nullptr;
    P_ESTIMATE_M estimateM = nullptr;
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++) {
        const char* extension = (const char*)glGetStringi(GL_EXTENSIONS, i);
        if (!strcmp(QCOM_F_EXTRAPOLATION_EXT_NAME, extension)) {
            SNN_LOGI("%s is supported", QCOM_F_EXTRAPOLATION_EXT_NAME);
            extrapolateTex = (P_EXTRAPOLATE_TEX)eglGetProcAddress(F_EXTRAPOLATION_FUN_NAME);
            break;
        }
    }
    for (GLint i = 0; i < n; i++) {
        const char* extension = (const char*)glGetStringi(GL_EXTENSIONS, i);
        if (!strcmp(QCOM_M_ESTIMATION_EXT_NAME, extension)) {
            SNN_LOGI("%s is supported", QCOM_M_ESTIMATION_EXT_NAME);
            estimateM = (P_ESTIMATE_M)eglGetProcAddress(F_M_ESTIMATION_FUN_NAME);
            break;
        }
    }
    if (test == 0 && !extrapolateTex) {
        SNN_LOGW("%d is not supported", QCOM_F_EXTRAPOLATION_EXT_NAME);
        return -1;
    }
    if (test >0 && !estimateM) {
        SNN_LOGW("%s is not supported", QCOM_M_ESTIMATION_EXT_NAME);
        return -1;
    }
    switch(test) {
        case 0:
            extrapolateFrames(context, framesDir, scaleFactor, extrapolateTex);
            break;
        case 1:
            estimateMotion(context, framesDir, estimateM);
            break;
        case 2:
            estimateMotionSynthetic(context, framesDir, estimateM, offset);
            break;
        case 3:
            testGSR(context, framesDir, scaleFactor);
            break;
        case 4:
            testOpticalFlowHS(context, framesDir, scaleFactor);
            break;
        case 5:
            testFSR(context, framesDir, scaleFactor);
            break;
        case 6:
            testWarp(context, framesDir, scaleFactor, estimateM);
            break;
        default:
            SNN_CHK(false);
    }

#endif
    return 0;
}