/* Copyright (C) 2020 - 2022 OPPO. All rights reserved.
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*        http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/
layout(OUTPUT_FORMAT, binding=0) writeonly uniform PRECISION image2DArray uOutImage;
layout(OUTPUT_FORMAT, binding=1) readonly uniform PRECISION image2DArray uInImage;
layout(binding = 3) readonly buffer weightBuffer{
    float data[];
} uWightBuffer;
layout(binding = 4) readonly buffer biasBuffer{
    float data[];
} uBiasBuffer;
layout(location = 5) uniform int uWidth;
layout(location = 6) uniform int uHeight;
layout(location = 7) uniform int activation;
//layout(binding= 8) writeonly buffer destBuffer{
//    float data[];
//} uOutBuffer;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
float relu(float i);
float sigmoid(float i);
float activeValue(int type, float v);

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    float res = 0.0f;
    for (int w = 0; w < uWidth; w+=4) 
    {
        vec4 color;
        vec4 weight;        
        int z = pos.z*4;
        vec4 color0 = imageLoad(uInImage, ivec3(w, 0, 0));
        vec4 color1 = imageLoad(uInImage, ivec3(w+1, 0, 0));
        vec4 color2 = imageLoad(uInImage, ivec3(w+2, 0, 0));
        vec4 color3 = imageLoad(uInImage, ivec3(w+3, 0, 0));
        weight.r = uWightBuffer.data[pos.y*uWidth+w];
        weight.g = uWightBuffer.data[pos.y*uWidth+w+1];
        weight.b = uWightBuffer.data[pos.y*uWidth+w+2];
        weight.a = uWightBuffer.data[pos.y*uWidth+w+3];    
        
        res += dot(vec4(color0.r, color1.r, color2.r, color3.r), weight);     
    }
    res += uBiasBuffer.data[pos.y];
    res = activeValue(activation, res);
    float test = float(uWidth);
    imageStore(uOutImage,ivec3(pos.y, 0, 0),vec4(res,0,0,0));
}

float relu(float i){
   if (i > 0.0){
       return i;
   } else {
       return 0.0;
   }
}

float sigmoid(float i){
    return 1.0 / (1.0 + exp(-i));
}

float activeValue(int type, float v){
    if (type == 0) {
        return (v);
    } else if (type == 1) {
        return relu(v);
    } else if (type == 2) {
        return sigmoid(v);
    } else if (type == 3){
        return tanh(v);
    } else {
        return v;
    }
}
