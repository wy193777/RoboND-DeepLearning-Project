# Project: Deep Learning Follow Me

[image_0]: ./docs/misc/sim_screenshot.png
[fcn]: ./docs/misc/fcn.png
[cnn]: ./docs/misc/cnn.png
[seg]: ./docs/misc/seg.png
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

In this project, we will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques we apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

![alt text][image_0]

## Network Architecture

![fcn][fcn]

Here we use Fully Convolutional Network (FCN) to deal semantic segmentation problem. The structure of a FCN is like the picture above. A convolutional neuro network usually used to classify images according to image content. To do semantic segmentation, the network have to perserve spatial information. That's why we have to use FCN.

An FCN is comprised of an encoder and decoder. The encoder portion is a convolution network that reduces to a deeper 1x1 convolution layer.

### Encoder

![cnn][cnn]

A cnn is connecting input images with multiple conbination of convoution layer, ReLu layer and pooling layer. Then at the end connection last layer to a fully connectioned layer to do classification job. But the fully connected layer at the end of CNN can only handle classification problems because it doesn't contains spatial information. To let spatial information preserved, we can connect another convolutional network instead of a fully connected layer.

### Decoder

On the decoder side of the network, we use some upsampling method to transfer previous layers to higher dimensions.

#### Bilinear Upsampling

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.

### 1x1 Convolution Layer

A 1x1 convolution simply maps an input pixel with all it's channels to an output pixel, not looking at anything around itself. It is often used to reduce the number of depth channels, since it is often very slow to multiply volumes with extremely large depths.

### Skip Connections

A cnn will looking closely at some images and lose the bigger picture as a result. Even if we were to decode the output of the encoder back to the original image size, some information has been lost. Skip connection is an easy way to retain those information. A skip layer will connect an encoder layer to decoder layer directly so the decoder layer will not miss 'the larger picture'.

So, in summary FCN is consisting of the following components:

- Encoder blocks: that will take inputs from previous layers, compress it down to a context losing in the way some of the resolution (the bigger picture).

- 1x1 Convolution block: that will reduce depth and capture the global context of the scene.

- Decoder blocks: that will take inputs from previous layers, decompress it, by up-sampling and adding inputs from previous encoder blocks through skip connections to recover some of the lost information hence do the precise segmentation.

- Skip connections: to retain some information from ecoder to decoder layer to make sure the network will not only focus on small features but also not missing the bigger picture.

- Softmax activation: normal convolution layer takes outputs from last decoder block and activate output pixels to indicate class and location of objects (semantic segmentation).

## Parameter Selection

### Learning Rate

Start with 0.001, and the result is good enough, so I didn't try other learning rate.

### Batch Size

This parameter is related to the size GPU memory size. When using 100 as batch size, TensorFlow will throw memory related exceptions. 80 works perfectly in on my GPU. 

### Steps Per Epoch

This variable is related to the batch size and sample size. batch_size * steps_per_epoch should equal to sample size. So I use `4370 // batch_size + 1` as my steps per epoch.

### Validation Steps

This variable is related to the batch size and validation data size. validation_steps * steps_per_epoch should equal to validation data size. So I use `1184 // batch_szie + 1` as my validation steps.

### Workers

This parameter seems doesn't have so much effect when train the model with GPU. My CPU has 8 cores and 16 logical process, I just choose 10 workers so I can also got some CPU for other task. CPU seem doesn't have much effect on GPU training. The CPU never been used more than 50% when a epoc is finished. Only 10% is used in most of training period.

## Result

The model can correctly identifiy humans from images. Even humans stand far away can also be correctly identified.

![seg][seg]

The final score of the model is 0.4574615522377517.

## Future Enhancements

The model only works with human because we only provided data related to human. To let the model works on other objects like cat, dog, car etc, we need to train the model with this kind of data.

In practice, we might want to train a FCN that could identify not only humans but also other physical objects so robot can avoid collisions.

## Note

The provided environment aren't working with GPU. And I cannot correctly set up the environment for specified TensorFlow version. After some struggle, I set it up using newest TensorFlow version. So my result is based on TensorFlow `1.11.0`.