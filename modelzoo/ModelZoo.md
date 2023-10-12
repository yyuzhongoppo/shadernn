![ShaderNN logo](../docs/images/logo.png)

## Model Zoo/Examples (Coming soon!):
  - image classification (Resnet18, MobileNetV2)
  - object detection (Yolov3-tiny)
  - image segmentation (U-Net)
  - image enchancement (ESPCN)
  - style transfer (StyleTransfer)

## Model Zoo models contents:

| Model          | Files                                   | Content                          | Usage                         |
|----------------|-----------------------------------------|----------------------------------|-------------------------------|
| Resnet18       | resnet18_cifar10.h5                     | Tensorflow model                 | Source to convert from        |
|                | resnet18_cifar10.tflite                 | Tensorflow model                 | Source to convert from        |
|                | resnet18_cifar10.param,                 | NCNN model structure             | Ground truth for SNN tests    |
|                | resnet18_cifar10.bin                    | NCNN weights                     |                               |
|                | resnet18_cifar10_layers.json,           | SNN model structure              | SNN tests, Demo app           |
|                | resnet18_cifar10_weights.bin            | SNN decoupled weights            |                               |
|                | resnet18_cifar10_0223.h5                | Tensorflow model                 | Source to convert from        |
|                | resnet18_cifar10_0223.param,            | NCNN model structure             | Ground truth for SNN tests    |
|                | resnet18_cifar10_0223.bin               | NCNN weights                     |                               |
|                | resnet18_cifar10_0223_layers.json,      | SNN model structure              | SNN tests                     |
|                | resnet18_cifar10_0223_weights.bin       | SNN decoupled weights            |                               |
|                | resnet18_cifar10_0223.json              | SNN model with weights           | SNN convolution test          |
|                |                                         |                                  |                               |
| MobileNetV2    | mobilenetv2_keras.h5                    | Tensorflow model                 | Source to convert from        |
|                | mobilenetv2_keras.param,                | NCNN model structure             | Ground truth for SNN tests    |
|                | mobilenetv2_keras.bin                   | NCNN weights                     |                               |
|                | mobilenetv2_keras_layers.json,          | SNN model structure              | SNN tests, Demo app           |
|                | mobilenetv2_keras_weights.bin           | SNN decoupled weights            |                               |
|                | mobilenetv2_pretrained_imagenet.h5      | Tensorflow model                 | Source to convert from        |
|                |                                         |                                  |                               |
|                | mobilenetv2_pretrained_imagenet.param,  | NCNN model structure             | Ground truth for SNN tests    |
|                | mobilenetv2_pretrained_imagenet.bin     | NCNN weights                     |                               |
|                | mobilenetv2_pretrained_imagenet.json    | SNN model with weights           | SNN tests                     |
|                |                                         |                                  |                               |
| Yolov3-tiny    | yolov3-tiny.h5                          | Tensorflow model                 | Source to convert from        |
|                | yolov3-tiny.tflite                      | Tensorflow model                 |                               |
|                | yolov3-tiny.param,                      | NCNN model structure             | Ground truth for SNN tests    |
|                | yolov3-tiny.bin                         | NCNN weights                     |                               |
|                | yolov3-tiny_layers.json,                | SNN model structure              | SNN tests, Demo app           |
|                | yolov3-tiny_weights.bin                 | SNN decoupled weights            |                               |
|                | yolov3_tiny_bb.h5                       | Tensorflow model                 | Source to convert from        |
|                | yolov3-tiny_bb.param,                   | NCNN model structure             | Ground truth for SNN tests    |
|                | yolov3-tiny_bb.bin                      | NCNN weights                     |                               | 
|                | yolov3_tiny_bb_layers.json,             | SNN model structure              | SNN tests                     |
|                | yolov3_tiny_bb_weights.bin              | SNN decoupled weights            |                               |
|                |                                         |                                  |                               |
| U-Net          | unet.h5                                 | Tensorflow model                 | Source to convert from        |
|                | unet.tflite                             | Tensorflow model                 |                               |
|                | unet.onnx                               | ONNX model                       |                               |
|                | unet.param                              | NCNN model structure             | Ground truth for SNN tests    |
|                | unet.bin                                | NCNN model structure             | Ground truth for SNN tests    |
|                | unet.json                               | SNN model with weights           | SNN tests                     |
|                | unet_layers.json, unet_weights.bin      | SNN model structure              |                               |
|                | unet_pretrained.h5                      | Tensorflow model                 | Source to convert from        |
|                | unet_pretrained.param,                  | NCNN model structure             | Ground truth for SNN tests    |
|                | unet_pretrained.bin                     | NCNN weights                     |                               | 
|                | unet_pretrained_layers.json,            | SNN model structure              | SNN tests                     |
|                | unet_pretrained_weights.bin             | SNN decoupled weights            |                               |
|                |                                         |                                  |                               |
| SpatialDenoise |                                         |                                  |                               |
|                | spatialDenoise.h5                       |                                  |                               |
|                | spatialDenoise.json                     | SNN model with weights           | SNN tests                     |
|                | spatialDenoise_layers.json,             | SNN model structure              |                               |
|                | spatialDenoise_weights.bin              | SNN decoupled weights            | Demo app                      | 
|                |                                         |                                  |                               |
| StyleTransfer  | candy-9_simplified.onnx                 | ONNX model                       |                               |
|                |                                         |                                  |                               |
|                |                                         |                                  |                               |
|                |                                         |                                  |                               |
|                | candy-9_simplified-opt.param,           | NCNN model structure             | Ground truth for SNN tests    |
|                | candy-9_simplified-opt.bin              | NCNN weights                     |                               |
|                | candy-9_simplified.json                 | SNN model with weights           | SNN tests, Demo app           |
|                | mosaic-9_simplified.json                | SNN model with weights           | Demo app                      |
|                | pointilism-9_simplified.json            | SNN model with weights           | Demo app                      |
|                | rain-princess-9_simplified.json         | SNN model with weights           | Demo app                      |
|                | udnie-9_simplified.json                 | SNN model with weights           | Demo app                      |
|                |                                         |                                  |                               |
| ESPCN          | ESPCN_2X_16_16_4.h5                     | Tensorflow model                 | Source to convert from        |
|                |                                         |                                  |                               |
|                | ESPCN_2X_16_16_4.json                   | SNN model with weights           | SNN tests                     |
|                | ESPCN_2X_16_16_4_layers.json,           | SNN model structure              |                               |
|                | ESPCN_2X_16_16_4_weights.bin            | SNN decoupled weights            |                               |