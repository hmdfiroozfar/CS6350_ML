# -*- coding: utf-8 -*-
"""Untitled50.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19-0llub9BiGnZ9KsQo9268RsHN1fhmqS
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Gaussian_SVM:
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        m, n = X_train.shape
        alphas = np.zeros(m)
        constraint = {'type': 'eq', 'fun': self._equality_constraint}
        bounds = [(0, self.C) for _ in range(m)]
        result = minimize(self._svm_dual_objective, alphas, args=(X_train, y_train),
                          method='SLSQP', bounds=bounds, constraints=constraint)
        alphas = result.x
        sv_indices = alphas > 1e-5
        self.support_vectors = X_train[sv_indices]
        self.support_vector_labels = y_train[sv_indices]
        self.alphas_sv = alphas[sv_indices]
        self.b = np.mean([y_k - np.sum(self.alphas_sv * self.support_vector_labels *
                                       np.array([self._gaussian_kernel(x_k, x_i) for x_i in self.support_vectors]))
                          for (x_k, y_k) in zip(self.support_vectors, self.support_vector_labels)])
        self.w = np.sum((self.alphas_sv * self.support_vector_labels)[:, np.newaxis] * self.support_vectors, axis=0)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, sv, sv_label in zip(self.alphas_sv, self.support_vectors, self.support_vector_labels):
                s += alpha * sv_label * self._gaussian_kernel(X[i], sv)
            y_pred[i] = s
        return np.sign(y_pred + self.b)

    def compute_error(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred != y)

    def _gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / self.gamma)

    def _gaussian_kernel_matrix(self, X):
        sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-sq_dists / self.gamma)

    def _svm_dual_objective(self, alphas, X, y):
        K = self._gaussian_kernel_matrix(X)
        return 0.5 * np.dot(alphas, np.dot(K, alphas)) - np.sum(alphas)

    def _equality_constraint(self, alphas):
        return np.dot(alphas, self.y_train)






C_values = [100/873, 500/873, 700/873]
gammas = [0.1, 0.5, 1, 5, 100]

path_to_train = 'train.csv'
path_to_test = 'test.csv'

train_data =pd.read_csv(path_to_train, header=None)
test_data = pd.read_csv(path_to_test, header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
y_train = np.where(y_train == 0, -1, 1)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
y_test = np.where(y_test == 0, -1, 1)

previous_support_vectors = None

for C in C_values:
    for gamma in gammas:
        svm = Gaussian_SVM(C, gamma)
        svm.train(X_train, y_train)

        train_error = svm.compute_error(X_train, y_train)
        test_error = svm.compute_error(X_test, y_test)

        print(f"\nGaussian Dual SVM for C: {C}, Gamma: {gamma},\n\t Train error: {train_error}, Test error: {test_error}")
        print(f"\tNumber of support vectors: {len(svm.support_vectors)}")
        print(f"\tWeight vector: {svm.w}, Bias: {svm.b}")

        if C == 500/873 and previous_support_vectors is not None:
            # Find overlapped support vectors
            overlap = np.sum(np.all(svm.support_vectors[:, None] == previous_support_vectors, axis=2), axis=1).astype(bool)
            num_overlapped = np.sum(overlap)
            print(f"\tNumber of overlapping support vectors with previous gamma: {num_overlapped}")

        previous_support_vectors = svm.support_vectors