## TODO: define the convolutional neural network architecture

import torch
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
        #Input shape (1,96,96)
        self.conv1 = nn.Conv2d(1, 32, 5) #Output: (32,92,92)
        self.pool1 = torch.nn.MaxPool2d(2,2) #Output: (32,46,46)
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(32, 64, 3) #Output: (64,44,44)
        self.pool2 = torch.nn.MaxPool2d(2,2) #Output: (64,22,22)
        self.drop2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, 3) #Output: (128,20,20)
        self.pool3 = torch.nn.MaxPool2d(2,2) #Output: (128,10,10)
        self.drop3 = nn.Dropout(p=0.3)
       
        self.fc1 = torch.nn.Linear(12800, 1000)#Output: 1000
        self.drop4 = nn.Dropout(p=0.4)
        self.fc2 = torch.nn.Linear(1000, 500)#Output: 500  
        self.drop5 = nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(500, 136)#Output: 136       
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #Convolutional Layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        #Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop4(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop5(x)
        x = self.fc3(x) #Linear activation in the last layer
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
