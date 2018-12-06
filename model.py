import numpy as np
import math
import torch.nn as nn
import torch

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fully connected neural network with one hidden layer
class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet, self).__init__()
        self.input_size = input_size

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 4,  padding = 3, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 4,  padding = 3, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 4,  padding = 3, stride = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(144, 12)
        self.fc2 = nn.Linear(12, 5)


    def forward(self, x):
        x = x.reshape((x.shape[0], 1, self.input_size, self.input_size))
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)

        return x