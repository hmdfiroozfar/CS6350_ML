import numpy as np
from math import log2

class DecisionTree:
    def __init__(self, train, labels, attributes, depth=-1, weights=None, impurity_measure='entropy'):
        self.impurity_functions = {
            'gini': self.gini_impurity,
            'entropy': self.entropy_impurity,
            'majority': self.majority_error
        }
        self.impurity_function = self.impurity_functions[impurity_measure]

        self.leaf = False
        self.label, num_values = self.determine_majority_label(labels, weights)

        if len(attributes) == 0 or num_values == 1 or depth == 0:
            self.leaf = True  
            return

        self.split_attribute, values = self.select_best_attribute(train, labels, attributes, weights)

        split_train, split_labels, split_weights = self.partition_data(train, labels, self.split_attribute, weights)
        self.sub_trees = {}
        attributes.remove(self.split_attribute)

        for value in split_train:
            self.sub_trees[value] = DecisionTree(split_train[value], split_labels[value], attributes, depth - 1, split_weights[value], impurity_measure)

        attributes.append(self.split_attribute)

    def predict(self, instance):
        if self.leaf:
            return self.label
        if instance[self.split_attribute] in self.sub_trees:
            return self.sub_trees[instance[self.split_attribute]].predict(instance)   
        return self.label   

    def select_best_attribute(self, train, labels, attributes, weights):
        initial_impurity = self.impurity_function(labels, weights)
        max_info_gain = -float('inf')
        best_attribute = None
        best_values = None

        for attribute in attributes:
            split_impurity, split_values = self.impurity_given_attribute(train, labels, attribute, weights)
            info_gain = initial_impurity - split_impurity

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attribute = attribute
                best_values = split_values

        return best_attribute, best_values

    def impurity_given_attribute(self, train, labels, attribute, weights=None):
        n = len(labels)
        if weights is None:
            weights = [1] * n

        partitioned_weights = {}
        partitioned_labels = {}
        total_weight = sum(weights)

        for idx, record in enumerate(train):
            attr_value = record[attribute]
            if attr_value not in partitioned_weights:
                partitioned_weights[attr_value] = []
                partitioned_labels[attr_value] = []

            partitioned_weights[attr_value].append(weights[idx])
            partitioned_labels[attr_value].append(labels[idx])

        impurity = 0
        for attr_value in partitioned_weights:
            weighted_impurity = self.impurity_function(partitioned_labels[attr_value], partitioned_weights[attr_value])
            impurity += (sum(partitioned_weights[attr_value]) / total_weight) * weighted_impurity

        return impurity, list(partitioned_weights.keys())

    # Impurity functions
    def gini_impurity(self, labels, weights=None):
        if weights is None:
            weights = [1] * len(labels)
        
        label_counts = {}
        total_weight = sum(weights)
        for idx, label in enumerate(labels):
            label_counts[label] = label_counts.get(label, 0) + weights[idx]

        impurity = 1 - sum([(count/total_weight)**2 for count in label_counts.values()])
        return impurity

    def entropy_impurity(self, labels, weights=None):
        if weights is None:
            weights = [1] * len(labels)
        
        label_counts = {}
        total_weight = sum(weights)
        for idx, label in enumerate(labels):
            label_counts[label] = label_counts.get(label, 0) + weights[idx]

        impurity = -sum([(count/total_weight) * log2(count/total_weight) for count in label_counts.values()])
        return impurity

    def majority_error(self, labels, weights=None):
        if weights is None:
            weights = [1] * len(labels)
        
        label_counts = {}
        for idx, label in enumerate(labels):
            label_counts[label] = label_counts.get(label, 0) + weights[idx]

        max_count = max(label_counts.values())
        return 1 - max_count / sum(weights)

    # Helper functions
    def determine_majority_label(self, labels, weights=None):
        if weights is None:
            weights = [1] * len(labels)
        
        label_counts = {}
        for idx, label in enumerate(labels):
            label_counts[label] = label_counts.get(label, 0) + weights[idx]

        majority_label = max(label_counts, key=label_counts.get)
        return majority_label, len(label_counts)

    def partition_data(self, train, labels, attribute, weights=None):
        n = len(labels)
        if weights is None:
            weights = [1] * n

        partitioned_train = {}
        partitioned_labels = {}
        partitioned_weights = {}
        
        for idx, record in enumerate(train):
            attr_value = record[attribute]
            if attr_value not in partitioned_train:
                partitioned_train[attr_value] = []
                partitioned_labels[attr_value] = []
                partitioned_weights[attr_value] = []

            partitioned_train[attr_value].append(record)
            partitioned_labels[attr_value].append(labels[idx])
            partitioned_weights[attr_value].append(weights[idx])

        return partitioned_train, partitioned_labels, partitioned_weights


import pandas as pd
import numpy as np

# Column names and types
columns = ['age_data', 'occupation', 'status', 'edu_level', 'has_default', 'account_balance', 
           'has_housing', 'has_loan', 'comm_type', 'day_num', 'month_name', 'call_duration',
           'num_campaign', 'days_passed', 'num_previous', 'outcome_prev', 'result']
data_types = ['num', 'cat', 'cat', 'cat', 'bool', 'num', 'bool', 'bool', 
              'cat', 'num', 'cat', 'num', 'num', 'num', 'num', 'cat', 'bool']
type_mapping = dict(zip(columns, data_types))

# Reading the datasets
train_data = pd.read_csv('train.csv', names=columns)
test_data = pd.read_csv('test.csv', names=columns)
print(train_data.head())

thresholds = {}
ProcessedTrain = pd.DataFrame()
ProcessedTest = pd.DataFrame()
for col in columns:
    if type_mapping[col] == 'num':
        median_value = train_data[col].median()
        thresholds[col] = median_value
        ProcessedTrain[col + '_is_gt_' + str(median_value)] = np.where(train_data[col] > median_value, "True", 'False')
        ProcessedTest[col + '_is_gt_' + str(median_value)] = np.where(test_data[col] > median_value, "True", 'False')
    else:
        ProcessedTrain[col] = train_data[col]
        ProcessedTest[col] = test_data[col]

TrainingSamples = ProcessedTrain.values.tolist()
TrainingLabels = [sample.pop() for sample in TrainingSamples]

TestingSamples = ProcessedTest.values.tolist()
TestingLabels = [sample.pop() for sample in TestingSamples]

feature_indices = list(range(len(columns) - 1))

def calculate_accuracy(tree_model, samples, actual_labels):
    predictions = [tree_model.predict(sample) for sample in samples]
    return (np.array(actual_labels) == np.array(predictions)).mean()

for impurity_type in ["entropy", "majority", "gini"]:
    for tree_depth in range(1, 16):
        tree_model = DecisionTree(TrainingSamples, TrainingLabels, feature_indices, depth=tree_depth, impurity_measure=impurity_type)
        print(f"For tree depth = {tree_depth}, training error = {1 - calculate_accuracy(tree_model, TrainingSamples, TrainingLabels)}")

for impurity_type in ['entropy', 'majority', "gini"]:
    for tree_depth in range(1, 16):
        tree_model = DecisionTree(TrainingSamples, TrainingLabels, feature_indices, depth=tree_depth, impurity_measure=impurity_type)
        print(f"For tree depth = {tree_depth}, test error = {1 - calculate_accuracy(tree_model, TestingSamples, TestingLabels)}")
