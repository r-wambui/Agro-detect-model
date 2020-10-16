---
id: index
title: Build a Simple Crop Disease Detection Model with PyTorch
sidebar_label: PyTorch Tutorial
---


## Build a Simple Crop Disease Detection Model with PyTorch

October 12, 2020

By Rose Wambui

### Introduction
In this tuitorial we will be creating a simple crop disease detector using PyTorch. We will use plant dataset that consists of 39 different classes of crop diseases with RGB images. We will leverage the power of Convolutional Neural Network(CNN)to achieve this.

### Prerequisites
 * Install PyTorch
 * Basic Understanding on Nueral Networks specifically in this case Convolutional Neural Network(CNN)

 

#### Install PyTorch
- Follow the guidelines on the [website](https://pytorch.org/) to install PyTorch. Based on your  *operating system*, *the package* and the *programming  laguage* you are given the command to run to install.

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/pytorch_install.png" />

<br> <br>

####  Convolutional Neural Network(CNN)
A CNN is a type of neural network which mainly include convolutional and pooling layers.

- Convolutional layer - contains a set of filters whose height and weight are smaller of the input image. These weights are then trained
- Pooling layer - Incorporated between two convolutional layers, a pooling layer reduces the number of parametrs and computatuon power by down-sampling the images through an activation function.
- Fully connected layer- Takes the end results of convolutional and pooling process and reaches a classification decision.

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/Architecture.png" />
<br>

Creating a CNN will involves the following:

```Step 1: Data loading and transformation```

```Step 2: Defining the CNN architecture```

```Step 3: Define loss and optimizer functions```

```Step 4: Training the model using the training set of data```

```Step 5: Validating the model using the test set```


NB  We will tackle this tutorial in a different format, where I will show the common errors I encountered while starting to learn PyTorch. 

#### Step 1: Data loading and transformation

##### 1.1 import our packages

``` 
import torch 
```


##### 1.2 Load data

- set up the data directory folder

```
data_dir = "" 
```
 Every image is inform of pixels which translate into arrays. PyTorch uses [PIL](https://pillow.readthedocs.io/en/stable/) - A python libarary for image processing.




