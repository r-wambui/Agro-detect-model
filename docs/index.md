---
id: index
title: Build a Simple Crop Disease Detection Model with PyTorch
sidebar_label: PyTorch Tutorial
layout: default
---


## Build a Simple Crop Disease Detection Model with PyTorch

October 12, 2020

By Rose Wambui

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/plant.jpg" />

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

{%- capture code -%}
import torch 
from torchvision import datasets, transforms, models

{%- endcapture -%}
{% include code.md code=code language='python' %}

##### 1.2 Load data

 Set up the data directory folder

```python
data_dir = "data" 
```
 Every image is inform of pixels which translate into arrays. PyTorch uses [PIL](https://pillow.readthedocs.io/en/stable/) - A python libarary for image processing.


 Pytorch uses [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) module to load datasets.The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision. We will use the [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) class to load our dataset.

 To load data using the **ImageFolder** data must be arranged in this format:

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

and **NOT** this format:

        root/xxx.png
        root/xxy.png
        root/123.png
        root/nsdf3.png

###### 1.3 Split the dataset int train and validation sets

It's a best practise to set aside validation data fro **inference** purposes

I have created a module [split_data](https://github.com/r-wambui/Agro-detect-model/raw/master/split_data.py) which splits any given image classification data into train and validation with a ration of 0.8:0.2.


```python
train_data = datasets.ImageFolder(data_dir + '/train')
val_data = datasets.ImageFolder(data_dir + '/val')

```

###### 1.4 Make the data Iterable

```python
dataiter = iter(train_data)
images, clases = dataiter
print(type(images))

```

The command above raises:

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/non_iterable.png" />

This means we can not iterate(meaning loop through) over the dataset. Pytorch use [DataLoader](https://pytorch.org/docs/stable/data.html) to make the dataset iterable

```python
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,)

```
```python
dataiter = iter(train_loader)
images, clases = dataiter.next()
print(type(images))

```

The code above raises:

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/tensor.png" />

 The [getitem](https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder) method of ImageFolder return unprocessed PIL image.  PyTorch uses tensors, since we will pass this data through pytorch models. We need to transform the image before using data loader. 

```python
train_transforms = transforms.Compose([transforms.ToTensor()])

val_transforms = transforms.Compose([transforms.ToTensor(),
])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
val_data = datasets.ImageFolder(data_dir + '/val', transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)

```
**batch_size** run 8 sample per iterations 

Run the data iter. This will raise a **runtime error**

<img src= "https://github.com/r-wambui/Agro-detect-model/raw/develop/static/img/runtime.png" />

Therefore we need to resize the images to the same shape before transforming it to a tensor

#### Data Transformation and Augumentation
{%- capture code -%}

train_transforms = transforms.Compose([transforms.RandomRotation(30), #data augumnetation
                                       transforms.RandomResizedCrop(224),#resize
                                       transforms.RandomHorizontalFlip(), #data augumnetation
                                       transforms.ToTensor(),
                                       ])

val_transforms = transforms.Compose([
                                      transforms.RandomResizedCrop(224), #resize
                                      transforms.ToTensor(),
                                      ])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
val_data = datasets.ImageFolder(data_dir + '/val', transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)


dataiter = iter(train_loader)
images, classes = dataiter.next()
print(type(images))
print(images.shape)
print(classes.shape)

{%- endcapture -%}
{% include code.md code=code language='python' %}

#### Data Visualization




### Model
 PyTorch [nn](https://pytorch.org/docs/stable/nn.html) module is used to build models.




