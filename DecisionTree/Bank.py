
from DecisionTree import *
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
train_data = pd.read_csv('ِData/Bank/train.csv', names=columns)
test_data = pd.read_csv('Data/Bank/test.csv', names=columns)

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

print("Training Error:")
for impurity_type in ["entropy", "majority", "gini"]:
    print(20*"--")
    print(f"Calculation with {impurity_type} as the impurity function:")
    print(20*"--")
    for tree_depth in range(1, 16):
        tree_model = DecisionTree(TrainingSamples, TrainingLabels, feature_indices, depth=tree_depth, impurity_measure=impurity_type)
        print(f"For tree depth = {tree_depth}, training error = {1 - calculate_accuracy(tree_model, TrainingSamples, TrainingLabels)}")

print("Testing Error:")
for impurity_type in ['entropy', 'majority', "gini"]:
    print(20*"--")
    print(f"Calculation with {impurity_type} as the impurity function:")
    print(20*"--")
    for tree_depth in range(1, 16):
        tree_model = DecisionTree(TrainingSamples, TrainingLabels, feature_indices, depth=tree_depth, impurity_measure=impurity_type)
        print(f"For tree depth = {tree_depth}, test error = {1 - calculate_accuracy(tree_model, TestingSamples, TestingLabels)}")



