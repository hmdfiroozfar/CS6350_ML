
import numpy as np
from random import shuffle, seed as random_seed


class Perceptron:
    def __init__(self, features, labels, epochs, shuffle_data=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random_seed(seed)
        self.weights = np.zeros(features.shape[1])
        self.weight_history = [self.weights.copy()]
        self.correct_counts = [1]
        self._train(features, labels, epochs, shuffle_data)

    def _train(self, features, labels, epochs, shuffle_data):
        num_samples = features.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            if shuffle_data:
                shuffle(indices)

            for index in indices:
                prediction = np.dot(features[index], self.weights)
                if labels[index] * prediction <= 0:
                    self.weights += labels[index] * features[index]
                    self.weight_history.append(self.weights.copy())
                    self.correct_counts.append(1)
                else:
                    self.correct_counts[-1] += 1

    def predict(self, data, perceptron_type='standard'):
        if perceptron_type == 'standard':
            return 2 * (data @ self.weight_history[-1] >= 0) - 1

        elif perceptron_type in ('voted', 'average'):
            weights_matrix = np.array(self.weight_history).T
            correct_counts = np.array(self.correct_counts)
            predictions = data @ weights_matrix
            if perceptron_type == 'voted':
                return 2 * ((2 * (predictions >= 0) - 1) @ correct_counts >= 0) - 1
            else:  # 'average'
                return 2 * (predictions @ correct_counts >= 0) - 1
        else:
            raise ValueError("Invalid perceptron type specified. Choose 'standard', 'voted', or 'average'.")

    def calculate_error(self, data, labels, perceptron_type='standard'):
        predictions = self.predict(data, perceptron_type=perceptron_type)
        return 1 - np.mean(predictions == labels)



def load_data(train_path, test_path):
  train = np.genfromtxt(train_path, delimiter=',')
  test = np.genfromtxt(test_path, delimiter=',')

  train_labels = 2 * train[:, -1] - 1
  test_labels = 2 * test[:, -1] - 1

  train[:, -1] = np.ones(train.shape[0])
  test[:, -1] = np.ones(test.shape[0])
  return train,train_labels,  test, test_labels

train_data,train_labels,  test_data, test_labels = load_data('Data/train.csv', 'Data/test.csv')

num_epochs = 10
r_seed = 20
# Standard Perceptron
standard_perceptron = Perceptron(train_data, train_labels, epochs = num_epochs, seed=r_seed)
print('Standard Perceptron:')
print('Train error:', standard_perceptron.calculate_error(train_data, train_labels))
print('Test error:', standard_perceptron.calculate_error(test_data, test_labels))
print('Learned weight vector:', standard_perceptron.weight_history[-1])

# Voted Perceptron
print('\nVoted Perceptron:')
print('Train error:', standard_perceptron.calculate_error(train_data, train_labels, perceptron_type='voted'))
print('Test error:', standard_perceptron.calculate_error(test_data, test_labels, perceptron_type='voted'))
distinct_weights = standard_perceptron.weight_history
counts = standard_perceptron.correct_counts
for weight, count in zip(distinct_weights, counts):
    print(f'Weight vector: {weight}, Count: {count}')

# Average Perceptron
print('\nAverage Perceptron:')
print('Train error:', standard_perceptron.calculate_error(train_data, train_labels, perceptron_type='average'))
print('Test error:', standard_perceptron.calculate_error(test_data, test_labels, perceptron_type='average'))
weights_average = np.average(np.array(standard_perceptron.weight_history), axis=0, weights=standard_perceptron.correct_counts)
print('Learned weight vector:', weights_average)