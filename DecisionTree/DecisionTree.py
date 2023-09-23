
import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, measure='entropy', max_depth=None):
        self.measure = measure
        self.max_depth = max_depth
        self.tree = None

    def fit(self, x, y):
        input_data = pd.concat([x, y], axis=1)
        self.tree = self.id3_alg(input_data, x.columns, self.max_depth)

    def id3_alg(self, data, attributes, depth):
        labels = data.iloc[:, -1].tolist()

    # If all labels are the same
        if len(set(labels)) == 1:
          return labels[0]

    # If depth is 0 or no attributes left
        if depth == 0 or len(attributes) == 0:
          most_common_label = max(set(labels), key=labels.count)
          return most_common_label

        best_attr = self.best_attribute(data, attributes)
        tree = {}
        tree[best_attr] = {}
        best_attr_values = data[best_attr].unique()
        for value in best_attr_values:
          subset = data[data[best_attr] == value].drop(columns=[best_attr])

          if depth:
            new_depth = depth - 1
          else:
            new_depth = None

          subtree = self.id3_alg(subset, subset.columns[:-1], new_depth)

          tree[best_attr][value] = subtree

        return tree

    def best_attribute(self, data, attributes):
        best_gain = -1
        best_attr = None

        for attr in attributes:
            gain = self.information_gain(data, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        return best_attr

    def information_gain(self, data, attribute):
        labels = data.iloc[:, -1].tolist()
        total_impurity = self.calculate_impurity(labels)

        attr_values = data[attribute].unique()
        total_weighted_impurity = 0

        for value in attr_values:
            subset_labels = data[data[attribute] == value].iloc[:, -1].tolist()
            weight = len(subset_labels) / len(data)
            impurity = self.calculate_impurity(subset_labels)
            total_weighted_impurity += weight * impurity

        return total_impurity - total_weighted_impurity

    def calculate_impurity(self, labels):
        if self.measure == 'entropy':
            return self.entropy(labels)
        elif self.measure == 'majority_error':
            return self.majority_error(labels)
        elif self.measure == 'gini':
            return self.gini_index(labels)
        else:
            raise ValueError(f"Unknown impurity measure: {self.measure}")

    def entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -sum(probabilities * np.log2(probabilities))
        return entropy

    def majority_error(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        majority_count = max(counts)
        me = 1 - majority_count / len(labels)
        return me

    def gini_index(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        gini = 1 - sum(probs**2)
        return gini

    def predict(self, data):
        results = []
        for _, row in data.iterrows():
            result = self.make_prediction(row, self.tree)
            results.append(result)
        return results

    def make_prediction(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        attribute = next(iter(tree))

        if row[attribute] in tree[attribute]:
            return self.make_prediction(row, tree[attribute][row[attribute]])
        else:
            labels = self.get_all_leaf_labels(tree[attribute])
            if not labels:
                raise ValueError("No leaf labels found in subtree values!")

            return max(labels, key=labels.count)

    def get_all_leaf_labels(self, subtree):
        if not isinstance(subtree, dict):
            return [subtree]

        labels = []
        for key in subtree:
          labels.extend(self.get_all_leaf_labels(subtree[key]))
        return labels

    @staticmethod
    def error_rate(y_true, y_pred):
        if len(y_true) != len(y_pred):
          raise ValueError("Input lists must have the same length")

        incorrect_predictions = 0
        for true, pred in zip(y_true, y_pred):
          if true != pred:
            incorrect_predictions += 1
        error = incorrect_predictions / len(y_true)
        return error
