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
        self.fc2 = nn.Linear(hidden_size, 8) 
        self.drop = nn.Dropout(0.01)
        self.fc3 = nn.Linear(8,num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = (self.fc3(self.drop(out)))
        return out

# Fully connected neural network with one hidden layer
class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 4,  padding = 3, stride = 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 4,  padding = 3, stride = 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 4,  padding = 3, stride = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(144, 48)
        self.drop = nn.Dropout(0.05)
        self.fc2 = nn.Linear(48, 5)

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, self.input_size, self.input_size))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.drop(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x