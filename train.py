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
from model import ConvNet


def train_torch(X_train,Y_train,X_test,Y_test): 
    print(X_train.shape)
    input_size = 16
    hidden_size = 8
    num_classes = 5
    learning_rate = 0.00025
    num_epochs = 12000
    model = NeuralNet(input_size, hidden_size, num_classes)
    #model = ConvNet(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    X_train = torch.from_numpy(X_train).float()
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test).float()
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.from_numpy(Y_test).long() 

    train_loss = []
    epochs = []
    for epoch in range(num_epochs):
        X_train = X_train
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < 0.001:
            break
        if (epoch+1) % 10 == 0:
            train_loss.append(loss.item())
            epochs.append(epoch+1)
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        #X_test = X_test.reshape(-1, 4*4)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted == Y_test).sum().item()
    plt.plot(epochs,train_loss)
    plt.title('Training Loss of 3-layer Network')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.show()

    print('Accuracy of the network: {} %'.format(100 * correct / total))
    return predicted
    




