# -*- coding: utf-8 -*-
"""Q2-e.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GzObfOMCYCF3iBOrStV4mA1DwgOrJeRC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt


def load_training_and_testing_data():
    train_features = []
    train_labels = []
    with open("train.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            train_features.append(item[:-1])
            train_labels.append(int(item[-1]))

    test_features = []
    test_labels = []
    with open("test.csv", "r") as f:
        for line in f:
            item = line.strip().split(",")
            test_features.append(item[:-1])
            test_labels.append(int(item[-1]))

    return np.asarray(train_features, dtype= float), np.asarray(train_labels, dtype= int), np.asarray(test_features, dtype= float), np.asarray(test_labels, dtype= int)


train_features, train_labels, test_features, test_labels = load_training_and_testing_data()


train_features_tensor = torch.from_numpy(train_features).type(torch.float32)
train_labels_tensor = torch.from_numpy(train_labels).type(torch.int64)
test_features_tensor = torch.from_numpy(test_features).type(torch.float32)
test_labels_tensor = torch.from_numpy(test_labels).type(torch.float64)


class CustomNeuralNetwork(nn.Module):
    def __init__(self, network_depth, input_size, hidden_layer_size, num_classes, activation_function, weight_initializer = nn.init.xavier_normal_):
        super().__init__()
        self.network_layers = []

        self.network_layers.append(nn.Linear(in_features=input_size, out_features=hidden_layer_size))
        weight_initializer(self.network_layers[-1].weight)

        for _ in range(network_depth-2):
            self.network_layers.append(nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size))
            weight_initializer(self.network_layers[-1].weight)

        self.network_layers.append(nn.Linear(in_features=hidden_layer_size, out_features=num_classes))
        weight_initializer(self.network_layers[-1].weight)

        self.activation_function = activation_function

        self.fc = nn.ModuleList(self.network_layers)

    def forward(self, x):
        for i in range(len(self.fc)-1):
            x = self.activation_function(self.fc[i](x))
        x = self.fc[-1](x)
        return x


print('"tanh" as activation function with "Xavier" initialization')


num_epochs = 100
lr = 1e-3
for network_depth in [3,5,9]:
    print('Network Depth = {}'.format(network_depth))
    for hidden_layer_size in [5,10,25,50, 100]:
        model = CustomNeuralNetwork(network_depth, 4, hidden_layer_size, 2, torch.tanh)
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            predicted_labels = model(train_features_tensor)
            loss = loss_criterion(predicted_labels, train_labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted_labels_train = model.forward(train_features_tensor)
        training_error = 1- (predicted_labels_train.max(axis = 1)[1] == train_labels_tensor).sum()/train_features_tensor.shape[0]

        predicted_labels_test = model.forward(test_features_tensor)
        testing_error = 1- (predicted_labels_test.max(axis = 1)[1] == test_labels_tensor).sum()/test_features_tensor.shape[0]
        print('Hidden Layer Size = {}: Training Error = {} and Testing Error = {}'.format(hidden_layer_size, training_error, testing_error))


print('"ReLU" as activation function with "He" initialization')


for network_depth in [3,5,9]:
    print('Network Depth = {}'.format(network_depth))
    for hidden_layer_size in [5,10,25,50, 100]:
        model = CustomNeuralNetwork(network_depth, 4, hidden_layer_size, 2, nn.ReLU(), weight_initializer = nn.init.kaiming_normal_)

        loss_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            predicted_labels = model(train_features_tensor)
            loss = loss_criterion(predicted_labels, train_labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted_labels_train = model.forward(train_features_tensor)
        training_error = 1- (predicted_labels_train.max(axis = 1)[1] == train_labels_tensor).sum()/train_features_tensor.shape[0]

        predicted_labels_test = model.forward(test_features_tensor)
        testing_error = 1- (predicted_labels_test.max(axis = 1)[1] == test_labels_tensor).sum()/test_features_tensor.shape[0]
        print('Hidden Layer Size = {}: Training Error = {} and Testing Error = {}'.format(hidden_layer_size, training_error, testing_error))