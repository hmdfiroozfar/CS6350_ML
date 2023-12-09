# -*- coding: utf-8 -*-
"""Q2-a,b,c.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GzObfOMCYCF3iBOrStV4mA1DwgOrJeRC
"""

import numpy as np

def learning_rate_schedule(time_step, initial_gamma, decay_rate):
    return initial_gamma / (1 + initial_gamma * time_step / decay_rate)

class NeuralNetwork:

    def __init__(self, hidden_layer_sizes, std_dev=1e-5):

        self.hidden_layers = hidden_layer_sizes
        self.num_hidden_layers = len(hidden_layer_sizes)

        self.weights = {}
        self.weights_gradient = {}

        self.biases = {}
        self.biases_gradient = {}

        self.layer_input = {}
        self.layer_input_gradient = {}

        self.layer_output = {}
        self.layer_output_gradient = {}

        self.std_dev = std_dev

    def train(self, X_train, y_train, epochs, regularization_strength, initial_gamma, decay_rate, batch_size=1):

        self.train_data = X_train
        self.train_labels = y_train

        self.num_samples = X_train.shape[0]
        self.num_features = X_train.shape[1]

        self.num_classes = len(np.unique(y_train))
        self.learning_rate = lambda t: learning_rate_schedule(t, initial_gamma, decay_rate)

        self.layer_sizes = [self.num_features] + self.hidden_layers + [self.num_classes]

        self.total_layers = len(self.layer_sizes)

        self.regularization_strength = regularization_strength

        for i in range(self.total_layers - 1):

            self.weights[i] = self.std_dev * np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
            self.biases[i] = np.zeros((1, self.layer_sizes[i+1]))

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(self.num_samples)
            shuffled_X = X_train[shuffled_indices, :]
            shuffled_y = y_train[shuffled_indices, :]

            for start in range(0, self.num_samples, batch_size):

                X_batch = shuffled_X[start:start+batch_size, :]
                y_batch = shuffled_y[start:start+batch_size, :]

                self.forward_pass(X_batch, y_batch)
                self.backward_pass(y_batch)

                for layer in range(self.total_layers - 1):

                    self.weights[layer] -= (1/batch_size) * self.learning_rate(epoch) * self.weights_gradient[layer] + 2 * self.learning_rate(epoch) * self.regularization_strength * self.weights[layer]
                    self.biases[layer] -= (1/batch_size) * self.learning_rate(epoch) * self.biases_gradient[layer]

    def forward_pass(self, X, y):

        batch_size = X.shape[0]

        self.layer_output[0] = X
        self.layer_input[0] = X

        for layer in range(self.total_layers - 1):

            self.layer_input[layer+1] = self.layer_output[layer] @ self.weights[layer] + self.biases[layer]

            if layer != self.total_layers - 2:
                self.layer_output[layer+1] = np.maximum(0, self.layer_input[layer+1]) # ReLU
            else:
                self.class_scores = self.layer_input[layer+1]

        adjusted_scores = self.class_scores - np.max(self.class_scores, axis=1, keepdims=True)
        exp_scores = np.exp(adjusted_scores)

        softmax_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        log_loss = -np.log(1e-30 + softmax_output[np.arange(batch_size), y.reshape(-1,)])

        regularization_loss = 0
        for layer in range(self.total_layers-1):
            regularization_loss += np.sum(self.weights[layer] * self.weights[layer])

        self.loss = np.sum(log_loss) / batch_size + self.regularization_strength * regularization_loss

        softmax_output[np.arange(batch_size), y.reshape(-1,)] -= 1
        self.layer_input_gradient[self.total_layers-1] = softmax_output

        return self.loss

    def backward_pass(self, y):

        batch_size = y.shape[0]

        for layer in range(self.total_layers-2, 0, -1):

            self.weights_gradient[layer] = self.layer_output[layer].T @ self.layer_input_gradient[layer+1]
            self.biases_gradient[layer] = np.sum(self.layer_input_gradient[layer+1], axis=0, keepdims=True)
            self.layer_output_gradient[layer] = self.layer_input_gradient[layer+1] @ self.weights[layer].T
            self.layer_input_gradient[layer] = self.layer_output_gradient[layer] * (self.layer_output[layer] > 0)

        self.weights_gradient[0] = self.layer_output[0].T @ self.layer_input_gradient[1]
        self.biases_gradient[0] = np.sum(self.layer_input_gradient[1], axis=0, keepdims=True)

    def predict(self, X):

        num_samples = X.shape[0]
        predictions = X

        for layer in range(self.total_layers-1):

            predictions = predictions @ self.weights[layer] + self.biases[layer]

            if layer != self.total_layers-2:
                predictions = np.maximum(0, predictions)

        return np.argmax(predictions, axis=1)

import numpy as np
import time
import random

# Loading the training and testing data
data = np.genfromtxt('train.csv', delimiter=",")
test = np.genfromtxt('test.csv', delimiter=",")

# Preparing the data
X_train = data[:, :-1]
X_test = test[:, :-1]
y_train = data[:, -1].astype(int).reshape(-1, 1)
y_test = test[:, -1].astype(int).reshape(-1, 1)

print("Initialize all the weights at random")

# Training and testing the model with random initialization
for layer_size in [5, 10, 25, 100]:
    model = NeuralNetwork([layer_size, layer_size], std_dev=1)
    model.train(X_train, y_train, epochs=5, regularization_strength=0.0, initial_gamma=0.01, decay_rate=0.01, batch_size=1)
    train_error = 1 - np.sum(model.predict(X_train).reshape(-1, 1) == y_train) / X_train.shape[0]
    test_error = 1 - np.sum(model.predict(X_test).reshape(-1, 1) == y_test) / X_test.shape[0]
    print('for width = {}, train error = {} and test error = {}'.format(layer_size, train_error, test_error))

print("Initialize all the weights with 0")

# Training and testing the model with zero initialization
for layer_size in [5, 10, 25, 100]:
    model = NeuralNetwork([layer_size, layer_size], std_dev=0)
    model.train(X_train, y_train, epochs=5, regularization_strength=0, initial_gamma=0.01, decay_rate=1, batch_size=10)
    train_error = 1 - np.sum(model.predict(X_train).reshape(-1, 1) == y_train) / X_train.shape[0]
    test_error = 1 - np.sum(model.predict(X_test).reshape(-1, 1) == y_test) / X_test.shape[0]
    print('for width = {}, train error = {} and test error = {}'.format(layer_size, train_error, test_error))