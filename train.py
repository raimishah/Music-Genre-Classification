import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from torch.autograd import Variable
from model import NeuralNet


def train_torch(X_train,Y_train,X_test,Y_test): 
    input_size = 16
    hidden_size = 8
    num_classes = 5
    learning_rate = 0.00001
    num_epochs = 45000
    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    X_train = torch.from_numpy(X_train).float()
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test).float()
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.from_numpy(Y_test).long() 

    for epoch in range(num_epochs):
        X_train = X_train.reshape(-1, 4*4)
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        X_test = X_test.reshape(-1, 4*4)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted == Y_test).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



