
![ShaderNN logo](images/logo.png)

  
##  Developer Integration Guide:  

The following guide describes the process and best practices to integrate ShaderNN in your application pipeline.  

In many cases during the whole execution of ShaderNN model inference, any model data is located on GPU and does not need to be copied to CPU or back.  
This reduces memory consumption and decreases latency of the model execution.  
ShaderNN supports both OpenGL and Vulkan engine backend. However, it must be built to support either one or both backends.  
Please refer to [Getting-Started.md](Getting-Started.md) for details. 
ShaderNN uses images, located on GPU as its inputs and usually outputs.  
To access GPU images, ShaderNN provides a simple wrapper API around native OpenGL or Vulkan API.  
Therefore, the best use case for ShaderNN is integration with graphics application that uses OpenGL or Vulkan natively.  

You need to choose the appropriate place in your application pipeline to make ShaderNN calls.  
ShaderNN creates its own pipeline, either graphics pipeline when using fragment shader or compute pipeline, when using compute shader.  
You must ensure that input images are ready when passing them to ShaderNN. That is, all rendering or other type of writing into input images  
must be completed and you might need to use appropriate synchronization mechanisms to ensure that.  
For example, in OpenGL use **glFinish()** command.  
In Vulkan, wait on appropriate fence.  
On the other hand, ShaderNN does synchronization by itself when produces output images. You can safely access output images right after ShaderNN call finishes.  

Below are the steps to integrate ShaderNN:  

### Create GPU context:

Example for OpenGL:  
```
auto context = snn::createGlContext();
```

Example for Vulkan:  
```
auto context = snn::createDefaultVulkanContext();
```

#### Vulkan initialization specific:

When using Vulkan backend, ShaderNN can initialize Vulkan using its own logic.  
However, application can (and most likely will) initialize Vulkan itself externally and pass to ShaderNN already initialized Vulkan data structures.  

Example for external Vulkan initialization:  
```
auto context = snn::createVulkanContext(
    vkInstance,
    vkPhysicalDevice,
    vkPhysicalDeviceMemoryProperties,
    vkDevice,
    vkCommandPool,
    vkQueue,
    vkQueueFamilyIndex
);
```
The Vulkan resources, outlined in the example will be shared between application and ShaderNN.  
If for some reason application does not want to share some of those resources, it must create this resource for ShaderNN purpose and manage it.  

### Initialize ShaderNN core:  

Usually input tensors for the deep learning model must be of certain fixed shape.  
For image processing inference models, usually the input tensors have 3 dimensions: width, height and logical number of channels.  
We also assume that the batch size is 1 for inference.  
On GPU, if logical number of channels is 4 or less, the input tensors are represented by 2 dimensional images.  
If logical number of channels is more than 4, the input tensors are represented by 3 dimensional images.  
Here we use term _logical_ to distinguish number of channels in a tensor from number of channels in the underlying images which can be only up to 4.  
ShaderNN requires that all input images must have 4 channels and must be in floating point format, either 32 bit or 16 bit.  
To initialize ShaderNN core, you need to pass in the input image(s) shape and color format.  
You also need to pass in backend type - Vulkan or OpenGL and some shader generation parameters. See [ShaderGenOptions](../core/inc/snn/layeroption.h) structure.  

Example for OpenGL:  
```
snn::dp::ShaderGenOptions options{};
snn::InferenceGraph::IODesc inputDesc {snn::ColorFormat::RGBA32F, width, height, depth, channels};
options.desiredInput.push_back(inputDesc);
options.compute = true; // Use compute shader
auto ic = snn::MixedInferenceCore::create(context, modelPath, options);
```

Example for Vulkan:  
```
snn::dp::ShaderGenOptions options{};
snn::InferenceGraph::IODesc inputDesc {snn::ColorFormat::RGBA32F, width, height, depth, channels};
options.desiredInput.push_back(inputDesc);
options.vulkan = true;
auto ic = snn::MixedInferenceCore::create(context, modelPath, options);
```

### Prepare input image:

For OpenGL ShaderNN uses OpenGL _textures_ to access images.  
ShaderNN requirement for OpenGL is that 2 dimensional images must be of **GL_TEXTURE_2D** type.  
3 dimensional images for OpenGL must be of **GL_TEXTURE_2D_ARRAY** type.  
For Vulkan all input images must be 3 dimensional, that is they must be of **VK_IMAGE_TYPE_3D** type.  
As we already mentioned, ShaderNN requires that all input images must have 4 channels and must be in floating point format, either 32 bit or 16 bit.  
ShaderNN was tested for the following GPU formats:  
For OpenGL: **GL_RGBA32F**, **GL_RGBA16F**  
For Vulkan: **VK_FORMAT_R32G32B32A32_SFLOAT**, **VK_FORMAT_R16G16B16A16_SFLOAT**  
Another important parameter for Vulkan images is layout. ShaderNN requires that all input Vulkan images must have **VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL** layout.  
The application must preprocess input images to be in ShaderNN and the model compatible format. If necessary, it must resize the image, convert the type, format and layout.  
_Note. If input image is located on CPU, ShaderNN has an API to convert it to GPU format, however the proper memory layout is required._   
Please see [ImageTexture](../core/inc/snn/imageTexture.h), [Image](../core/inc/snn/image.h) classes.  

### Prepare output image:

If output image is needed as a result of model inference, the application must prepare it and pass it to ShaderNN.  
The requirements for the output image is the same as for input image, except for Vulkan layout requirement.  
For Vulkan backend, output image must be in one of those formats:  
**VK_IMAGE_LAYOUT_UNDEFINED**, **VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL**, **VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL**, **VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL**.  
_Note: When initializing the empty output image, you can use ShaderNN API to do all low-level initialization work._
For classification and some object detection models, ShaderNN generated output is not an image, but special CPU data structures (see below).  
In this case you don't need to prepare output image and pass it to ShaderNN. 

### Run inference session:

Example for OpenGL:  
```
snn::MixedInferenceCore::RunParameters rp{};

snn::ImageTextureGLArray inputImageTexs;
inputImageTexs.allocate(1);
inputImageTexs[0].reset({widthIn, heightIn, depthIn, 1}, snn::ColorFormat::RGBA32F);
inputImageTexs[0].texture(0)->attach(textureTarget, textureId);
rp.inputImages = inputImageTexs;

snn::ImageTextureGLArray outputImageTexs;
outputImageTexs.allocate(1);
outputImageTexs[0].reset({widthOut, heightOut, depthOut, 1}, snn::ColorFormat::RGBA32F);
rp.outputImages = outputImageTexs;

ic->run(rp);
```

Example for Vulkan:  
```
snn::MixedInferenceCore::RunParameters rp{};

snn::ImageTextureVulkanArray inputImageTexs{snn::ImageTextureVulkanAllocator(context)};
inputImageTexs.allocate(1);
inputImageTexs[0].reset({widthIn, heightIn, depthIn, 1}, snn::ColorFormat::RGBA32F);
inputImageTexs[0].attach(vkImage, vkImageView, vkImageLayout);
rp.inputImages = inputImageTexs;

snn::ImageTextureVulkanArray outputImage{snn::ImageTextureVulkanAllocator(context)};
outputImageTexs.allocate(1);
outputImageTexs[0].reset({widthOut, heightOut, depthOut, 1}, snn::ColorFormat::RGBA32F);
rp.outputImages = outputImageTexs;

ic->run(rp);
```

#### Obtaining classification and image detection results:

Currently, ShaderNN records classifier output and object detection output in [RunParameters::modelOutput](../core/inc/snn/core.h#L82) member.  
Classifier output is recorded in [SNNModelOutput::classifierOutput](../core/inc/snn/snn.h#L113) member and object output is recorded in [SNNModelOutput::detectionOutput](../core/inc/snn/snn.h#L114) member.  