import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, C, gamma_0=0.01, a=None, epochs=100):
        self.epochs = epochs
        self.C = C
        self.gamma_0 = gamma_0
        self.a = a
        self.w = None
        self.b = 0

    def objective_function(self, X, y):
        hinge_losses = [max(0, 1 - y[i] * (np.dot(self.w, x) + self.b)) for i, x in enumerate(X)]
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_losses)

    def fit_svm(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)

        objective_values = []
        for epoch in range(self.epochs):
            shuffled_idx = np.random.permutation(len(y_train))
            X_train = X_train[shuffled_idx]
            y_train = y_train[shuffled_idx]
            for i in range(n_samples):
                if self.a is not None:
                  gamma_t = self.gamma_0 / (1 + (self.gamma_0 / self.a) * epoch)
                else:
                  gamma_t = self.gamma_0 / (1 + epoch)

                if y_train[i] * (np.dot(self.w, X_train[i]) + self.b) < 1:
                    self.w += gamma_t * (y_train[i] * X_train[i] - (2 * self.C * self.w / n_samples))
                    self.b += gamma_t * y_train[i] * self.C
                else:
                    self.w -= gamma_t * (2 * self.C * self.w / n_samples)

            objective_values.append(self.objective_function(X_train, y_train))
        return objective_values

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


C_values = [100/873, 500/873, 700/873]
gamma_list = [0.01, 0.05, 0.1]  
a_list = [0.01, 0.05, 0.1]     

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

print('*************PART A**************')
weights_list = []
bias_list = []
train_error_list = []
test_error_list = []

for C in C_values:
  for a in a_list:
    for gamma_0 in gamma_list:
        svm = SVM(C=C, gamma_0=gamma_0, a=a)
        objective_values = svm.fit_svm(X_train, y_train, X_test, y_test)
        train_error = 1 - svm.score(X_train, y_train)
        test_error = 1 - svm.score(X_test, y_test)
        print(f'For C = {C}, gamma_0 = {gamma_0} and a = {a},\n\t Training error = {train_error:.4f}, Test error = {test_error:.4f}')
        plt.plot(objective_values, label=f'C={C}')
        plt.title(f'SVM Primal for C={C}, gamma_0 = {gamma_0}, a = {a}')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.legend()
        plt.show()

        if gamma_0 == 0.01 and a == 0.01:
          weights = svm.w
          bias = svm.b
          weights_list.append(weights)
          bias_list.append(bias)
          train_error_list.append(train_error)
          test_error_list.append(test_error)
          print(f'For C = {C}:\n\tWeights: {weights}, bias:{bias},\n\tTrain Error: {train_error}, Test Error: {test_error}')


print('**************(PART B)**************')

weights_list_2 = []
bias_list_2 = []
train_error_list_2 = []
test_error_list_2 = []

for C in C_values:
  for gamma_0 in gamma_list:
    svm = SVM(C=C, gamma_0=gamma_0, epochs=100, a = None)
    objective_values = svm.fit_svm(X_train, y_train, X_test, y_test)
    train_error = 1 - svm.score(X_train, y_train)
    test_error = 1 - svm.score(X_test, y_test)
    print(f'For C = {C} and gamma_0 = {gamma_0},\n\t Training error = {train_error:.4f}, Test error = {test_error:.4f}')
    plt.plot(objective_values, label=f'C={C}')
    plt.title(f'SVM Primal for C={C} and gamma_0 = {gamma_0}')
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.show()

    if gamma_0 == 0.01:
          weights = svm.w
          bias = svm.b
          weights_list_2.append(weights)
          bias_list_2.append(bias)
          train_error_list_2.append(train_error)
          test_error_list_2.append(test_error)
          print(f'For C = {C}:\n\tWeights: {weights}, bias:{bias},\n\tTrain Error: {train_error}, Test Error: {test_error}')


print('**************(PART C)**************')
for i in range (len(C_values)):
    C = C_values[i]

    weight_1 = weights_list[i]
    weight_2 = weights_list_2[i]
    bias_1 = bias_list[i]
    bias_2 = bias_list_2[i]

    tr_err_1 = train_error_list[i]
    tr_err_2 = train_error_list_2[i]
    tst_err_1 = test_error_list[i]
    tst_err_2 = test_error_list_2[i]

    weight_diff = np.linalg.norm(weight_1 - weight_2)
    bias_diff = abs(bias_1 - bias_2)
    train_error_diff = abs(tr_err_1 - tr_err_1)
    test_error_diff = abs(tst_err_1 - tst_err_2)
    print(f"\nFor C={C}:")
    print(f"Weights Difference: {weight_diff}")
    print(f"Bias Difference: {bias_diff}")
    print(f"Train Error Difference: {train_error_diff}")
    print(f"Test Error Difference: {test_error_diff}")