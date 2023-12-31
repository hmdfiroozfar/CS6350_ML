# -*- coding: utf-8 -*-
"""Adaboost_Q2_a.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qqopG0L7Rd9ccBh6M64WI1fNe6D9Al96
"""

import pandas as pd
import numpy as np
from numpy import log2, log, sqrt
import matplotlib.pyplot as plt
from random import sample
from Adaboost import *


COLUMN_NAMES = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

COLUMN_TYPES = ['numeric', 'categorical', 'categorical', 'categorical', 'binary', 'numeric',
                      'binary', 'binary', 'categorical', 'numeric', 'categorical', 'numeric',
                      'numeric', 'numeric', 'numeric', 'categorical', 'binary']

COLUMN_DICT = dict(zip(COLUMN_NAMES, COLUMN_TYPES))


def process_data(train_path, test_path, column_dict, column_names):
    train_data = pd.read_csv(train_path, names=column_names)
    test_data = pd.read_csv(test_path, names=column_names)

    median_dict = {}
    processed_train = pd.DataFrame()
    processed_test = pd.DataFrame()
    for column in column_names:
        if column_dict[column] == 'numeric':
            median_val = train_data[column].median()
            median_dict[column] = median_val
            processed_train[column + '>' + str(median_val)] = np.where(train_data[column] > median_val, 'yes', 'no')
            processed_test[column + '>' + str(median_val)] = np.where(test_data[column] > median_val, 'yes', 'no')
        else:
            processed_train[column] = train_data[column]
            processed_test[column] = test_data[column]

    processed_train_values = [list(processed_train.loc[i]) for i in range(len(processed_train))]
    processed_test_values = [list(processed_test.loc[i]) for i in range(len(processed_test))]

    return processed_train_values, processed_test_values


def convert_labels(labels):
    return [1 if label == 'yes' else -1 for label in labels]


def plot_errors(train_errors, test_errors, title):
    plt.plot(train_errors, color='blue', label="Train")
    plt.plot(test_errors, color='red', label="Test Error")
    plt.title(title, color='black')
    plt.legend()
    plt.show()


def main(num_iters, depth, num_trees):
    processed_train, processed_test = process_data('Data/train.csv', 'Data/test.csv', COLUMN_DICT, COLUMN_NAMES)

    training_data = [sample[:-1] for sample in processed_train]
    training_labels = convert_labels([sample[-1] for sample in processed_train])

    testing_data = [sample[:-1] for sample in processed_test]
    testing_labels = convert_labels([sample[-1] for sample in processed_test])

    ada_model = AdaBoost(training_data, training_labels, num_iterations = num_iters, tree_depth = depth)

    train_errors = ada_model.compute_overall_error(training_data, training_labels, num_trees = num_trees)
    test_errors = ada_model.compute_overall_error(testing_data, testing_labels, num_trees = num_trees)

    plot_errors(train_errors, test_errors, "Errors vs iterations")

    individual_tree_train_errors = []
    individual_tree_test_errors = []
    for tree in ada_model.decision_trees:
        individual_tree_train_error = sum(1 for i in range(len(training_data)) if tree.predict(training_data[i]) != training_labels[i]) / len(training_data)
        individual_tree_test_error = sum(1 for i in range(len(testing_data)) if tree.predict(testing_data[i]) != testing_labels[i]) / len(testing_data)

        individual_tree_train_errors.append(individual_tree_train_error)
        individual_tree_test_errors.append(individual_tree_test_error)

    plot_errors(individual_tree_train_errors, individual_tree_test_errors, "Decision Stumps Errors vs iterations")

num_iters = 500
depth = 3
num_trees = 500

if __name__ == "__main__":
    main(num_iters, depth, num_trees)