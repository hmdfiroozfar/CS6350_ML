

from DecisionTree import DecisionTree
import pandas as pd

column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

train_data = pd.read_csv("Data/Car/train.csv", names=column_names)
test_data = pd.read_csv("Data/Car/test.csv", names=column_names)


X_train = train_data.drop(columns=["label"])
y_train = train_data["label"]

X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]

depth_values = list(range(1, 7))
impurity_methods = ['entropy', 'majority_error', 'gini']
results = {'impurity method': [], 'max tree depth': [], 'train error': [], 'test error': []}

for impurity_method in impurity_methods:
    for depth in depth_values:
        results['impurity method'].append(impurity_method)
        results['max tree depth'].append(depth)

        dt_model = DecisionTree(measure = impurity_method, max_depth = depth)
        dt_model.fit(X_train, y_train)

        y_test_pred = dt_model.predict(X_test)
        y_train_pred = dt_model.predict(X_train)

        train_error = dt_model.error_rate(y_train.to_list(), y_train_pred)
        test_error = dt_model.error_rate(y_test.to_list(), y_test_pred)

        results['train error'].append(train_error)
        results['test error'].append(test_error)


output = pd.DataFrame(results)
print(output)
