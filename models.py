## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1  = nn.Conv2d(1,    16,   5, padding=2)
        self.conv2  = nn.Conv2d(16,   16,   5, padding=2)
        self.conv3  = nn.Conv2d(16,   32,   3, padding=1)
        self.conv4  = nn.Conv2d(32,   32,   3, padding=1)
        self.conv5  = nn.Conv2d(32,   64,   3, padding=1)
        self.conv6  = nn.Conv2d(64,   64,   3, padding=1)
        self.conv7  = nn.Conv2d(64,   128,  3, padding=1)
        self.conv8  = nn.Conv2d(128,  128,  3, padding=1)
        
        self.fc1 = nn.Linear(4*4*128, 1000)
        self.fc2 = nn.Linear(1000,    136)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv1.weight.data.uniform_(-0.03, 0.03)
        self.conv2.weight.data.uniform_(-0.03, 0.03)
        self.conv3.weight.data.uniform_(-0.03, 0.03)
        self.conv4.weight.data.uniform_(-0.03, 0.03)
        self.conv5.weight.data.uniform_(-0.03, 0.03)
        self.conv6.weight.data.uniform_(-0.03, 0.03)
        self.conv7.weight.data.uniform_(-0.03, 0.03)
        self.conv8.weight.data.uniform_(-0.03, 0.03)
        
        self.fc1.weight.data.uniform_(-0.2, 0.2)
        self.fc2.weight.data.uniform_(-0.2, 0.2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool4(F.relu(self.conv7(x)))
        x = self.pool5(F.relu(self.conv8(x)))
        
        x = x.view(x.size(0), -1)
       
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
            
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
        
#         ## TODO: Define all the layers of this CNN, the only requirements are:
#         ## 1. This network takes in a square (same width and height), grayscale image as input
#         ## 2. It ends with a linear layer that represents the keypoints
#         ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
#         # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
#         # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1  = nn.Conv2d(1,    16,   3, padding=1)
#         self.conv2  = nn.Conv2d(16,   32,   3, padding=1)
#         self.conv3  = nn.Conv2d(32,   64,   3, padding=1)
#         self.conv4  = nn.Conv2d(64,   128,  3, padding=1)
#         self.conv5  = nn.Conv2d(128,  256,  3, padding=1)


#         self.fc1 = nn.Linear(4*4*256, 1000)
#         self.fc2 = nn.Linear(1000,    1000)
#         self.fc3 = nn.Linear(1000,    136)
        
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.pool5 = nn.MaxPool2d(2, 2)
        
#         self.conv1.weight.data.uniform_(-0.03, 0.03)
#         self.conv2.weight.data.uniform_(-0.03, 0.03)
#         self.conv3.weight.data.uniform_(-0.03, 0.03)
#         self.conv4.weight.data.uniform_(-0.03, 0.03)
#         self.conv5.weight.data.uniform_(-0.03, 0.03)
        
#         self.fc1.weight.data.uniform_(-0.2, 0.2)
#         self.fc2.weight.data.uniform_(-0.2, 0.2)
#         self.fc3.weight.data.uniform_(-0.2, 0.2)
        
#         ## Note that among the layers to add, consider including:
#         # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
#     def forward(self, x):
#         ## TODO: Define the feedforward behavior of this model
#         ## x is the input image and, as an example, here you may choose to include a pool/conv step:
#         ## x = self.pool(F.relu(self.conv1(x)))
        
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.pool5(F.relu(self.conv5(x)))
        
#         x = x.view(x.size(0), -1)
        
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
            
#         # a modified x, having gone through all the layers of your model, should be returned
#         return x
    
  