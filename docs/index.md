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
{:toc}
In this tuitorial we will be creating a simple crop disease detector using PyTorch. We will use plant dataset that consists of 39 different classes of crop diseases with RGB images. We will leverage the power of Convolutional Neural Network(CNN)to achieve this.

### Prerequisites
{:toc}
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


####  Step 2: Model architecture

PyTorch [nn](https://pytorch.org/docs/stable/nn.html) module is used to build models.

When creating CNN, understanding the output dimensions after every connvolutional and pooling layers is important.

#### Calculate output dimensions
This is the formula to calculate dimensions through convolutional layer

<img src="https://latex.codecogs.com/gif.latex?O=(\frac{W-K+2P}{S}+1)">

Where;

    O - The ouptut height/width
    W - The input height/width
    K - The kernel size
    P - Padding
    S - Stride

This is used to calculate dimensions after a max pool layer

<img src="https://latex.codecogs.com/gif.latex?O=(\frac{W}{K})">

We will create a **sample CNN model of this architecture**:

  - *2 convolutional layers*
  - *2 max pooling layers* 
  - *1 fully connected layer*


Input_image shape(RGB) = 224, 224, 3

**1st convolutional layer**

- W = 224
- K = 3
- P = 0
- S = 1

<img src="https://latex.codecogs.com/gif.latex?O=(\frac{224-3+2*0}{1}+1)">

(224 - 3 ) + 1 = 222 

the ouput will be 222 x 222 x 16 : **Note** 16 in the channel/color dimensions we have selected.



**1st max pool layer**

shape = 222 x 222 x 16
k = 2

222 / 2 = 111
 
 
The output image will be 111 x 111 x 16 (the color channel does not change after a max pool layer)
 
 
 
**2nd Convolutional Layer** 
 
 - W = 111 
 - K = 3
 - P = 1
 - S = 2
 
 <img src="https://latex.codecogs.com/gif.latex?O=(\frac{111-4+2*1}{2}+1)">

 
 (111 - 3 + 2*1)/2 + 1 = 56 
 
  output image 56x 56 x 32
  
  
 
 **2nd maxpooling layer**
 
 56/2 = 28
 
 output image 28 x 28 x 32
 
 **Fully connected layer**
 
In the fully connected layer you pass a flattened image and the number of output clases required in this case is 39.


{%- capture code -%}
import torch.nn as nn
import numpy as np

class CropDetectCNN(nn.Module):
    # initialize the class and the parameters
    def __init__(self):
        super(CropDetectCNN, self).__init__()
        
        # convolutional layer 1 & max pool layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2))
        
        # convolutional layer 2 & max pool layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.MaxPool2d(kernel_size=2))
        
        #Fully connected layer
        self.fc = nn.Linear(32*28*28, 39)
     

    
    
    # Feed forward the network
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CropDetectCNN()
{%- endcapture -%}
{% include code.md code=code language='python' %}


### Step 3: Loss and Optimizer
Loss determines how far the model deviates from predicitng true values while Optimizer is the function used to change the attributes/parameter of the neural networks such as weights and learning rate.

These functions are dependant on the type of machine learning problem you are trying to solve. In our case we are dealing with multi-class classification. You can research more on loss and optimization in neural networks.

For this case we'll use Cross Entropy Loss and Stochastic Gredient Descent(SGD)

{%- capture code -%}
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
{%- endcapture -%}
{% include code.md code=code language='python' %}

Image analysis require very high processing power, Therefore you can leverage free GPUs in the market. PyTorch uses CUDA to enable developers to run their products on gpu enable environment. 

{%- capture code -%}
# run on GPU if available else run on a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
{%- endcapture -%}
{% include code.md code=code language='python' %}

#### Step 4: Model Training

{%- capture code -%}
epochs = 1 #run more iterations

for epoch in range(epochs):
    running_loss = 0
    for images, classes in train_loader:

        # Use GPU 
        images, classes = images.to(device), classes.to(device)
        
        # clears old gradients from the last step
        optimizer.zero_grad()
        
        # train the images
        outputs = model(images)
        
        #calculate the loss given the ouputs and the classes
        loss = criterion(outputs, classes)
        
        # compute the loss of every parameter
        loss.backward()
        
        # apply the optimizer and it's parameters
        optimizer.step()
        
        #update the loss
        running_loss += loss.item()
        
    else:
        validation_loss = 0
        accuracy = 0
        
        # to make the model run faster we are using the gradients on the train
        with torch.no_grad():
            # specify that this is validation and not training
            model.eval()
            for images, classes in val_loader:
                
                # Use GPU
                images, classes = images.to(device), classes.to(device)
                
                # validate the images
                outputs = model(images)
                
                # compute validation loss
                loss = criterion(outputs, classes)
                
                #update loss
                validation_loss += loss.item()
                
                # get the exponential of the outputs
                ps = torch.exp(outputs)
                
                #Returns the k largest elements of the given input tensor along a given dimension.
                top_p, top_class = ps.topk(1, dim=1)
                
                # reshape the tensor
                equals = top_class == classes.view(*top_class.shape)
                
                # calculate the accuracy.
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        # change the mode to train for next epochs
        model.train()

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
              "Valid Loss: {:.3f}.. ".format(validation_loss/len(val_loader)),
              "Valid Accuracy: {:.3f}".format(accuracy/len(val_loader)))

{%- endcapture -%}
{% include code.md code=code language='python' %}