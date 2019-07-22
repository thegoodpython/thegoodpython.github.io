---
layout: post
title: "Image prediction"
author: rishi
categories: [tensorflow, keras]
post-image: assets/images/mobilenet.png
publish: true
---

In this tutorial, we will see how to write a program that uses a neural network that has already been trained.  
We will use the neural network to tell us what the image contains.  

![MobileNet block diagram]({{ site.baseurl }}/assets/images/mobilenet.png)

The neural network that we will be using is called the MobileNet.  

> MobileNet is a neural network architecture, which is suitable for mobile and embedded devices (hence the name, MobileNet). It is a light weight deep neural network

It has been trained on the ImageNet dataset.  

> ImageNet is a large database with more than 14 million images of more than 20 thousand categories

We will be using MobileNet because it is extremely lightweight (14 MB) and therefore can be downloaded quite quickly.  

For complete code: [Click here](https://nbviewer.jupyter.org/github/rishi93/thegoodpython_notebooks/blob/master/image_prediction.ipynb)

# First we import the libraries
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
```

The Tensorflow library helps to introduce neural networks in our Python program  
The PIL (Python Imaging Library) allows to handle images in our Python program  
The Numpy library allows us to handle multidimensional arrays in our Python program  
The Matplotlib library allows to plot graphs and display images in our Python program  

# Load the image into the program
```python
image = PIL.Image.open('images/frog.jpg')
image = image.resize((224, 224))
image = np.array(image)
```

First we load the image using the PIL library  
We resize the image. MobileNet only accepts images of height 224 and width 224  
We convert the image to a 2D array format  

# Download the Neural Network
```python
model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet)
```

The Keras API provides an easy way to download the MobileNet neural network from the internet.  
The *include_top=True* means that the top part of the MobileNet is also going to be downloaded.  
The top part is what helps to categorize the image correctly (We will look into this in more detail later).  
The *weights='imagenet'* means that the weights of the neural network are the ones obtained by training the network on the ImageNet dataset.  

# Preprocess the input
```python
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
```

MobileNet takes image input in a certain format.  
A normal image has pixel values between 0 to 255.  
MobileNet accepts input images with pixel ranges between -1 and +1.  
The *preprocess_input* function helps us scale our values to the -1 to +1 range.  

# Create a batch with our input image
```python
image = image[tf.newaxis, :]
```

![Batch of images]({{ site.baseurl }}/assets/images/newaxis.png)

Neural Networks take input in the form of batches. A batch is a group of inputs.  
Since we have only one input (one image). Our batch will contain only one image.  
If we had multiple images, we would have multiple images within one batch.  
If we print the shape of *image* variable now,  
We notice:
```python
print(image.shape)
# => (1, 224, 224, 3)
```
Our input image is of dimensions (224, 224, 3). 224 pixels in height and width and 3 channels (Red, Green, Blue).
The 1 means that we have 1 input image in this batch.  

# Input the image to the Neural Network
```python
predictions = model.predict(image)
```

We use the neural network to make predictions on our input batch of one image. 

# Decode the predictions
```python
top5 = tf.keras.predictions.mobilenet_v2.decode_predictions(predictions)[0]
```

Neural networks take in input in the form of numbers.  
An image of a frog, may look like a frog to a human. But to a machine, it's a 2D array of numbers.  
Similarly, Neural Networks output is also in the form of numbers.  
These numbers represent probabilities.  
ImageNet has 1000's of categories, so the probabilities mean that the image is one of these categories.  
To decode these predictions, we use the *decode_predictions* function.  

# Print the predictions
```python
for num, name, score in top5:
    print(num, name, score*100)
```