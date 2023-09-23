from DecisionTree import DecisionTree
import pandas as pd

def handle_unknown_values(train, test, unknown_is_value="no"):
    if unknown_is_value != "yes":
        for column in train.columns:
            mode_value = train[train[column] != "unknown"][column].mode()[0]
            train[column] = train[column].replace("unknown", mode_value)
            test[column] = test[column].replace("unknown", mode_value)
    return train, test

def run_decision_tree_model(train_path, test_path, max_depth_value, unknown_is_value):
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

    train_data = pd.read_csv(train_path, names=column_names)
    test_data = pd.read_csv(test_path, names=column_names)

    train_data, test_date = handle_unknown_values(train_data, test_data, unknown_is_value = unknown_is_value)
    

    for column in train_data.select_dtypes(include=['number']):
        median_value = train_data[column].median()
        train_data[column] = (train_data[column] > median_value).astype(int)

    for column in test_data.select_dtypes(include=['number']):
        median_value = test_data[column].median()
        test_data[column] = (test_data[column] > median_value).astype(int)

    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]

    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]

    depth_values = list(range(1, max_depth_value+1))
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
    return output

train = "Data/Bank/train.csv"
test = "Data/Bank/test.csv"

output = run_decision_tree_model(train, test, max_depth_value = 16, unknown_is_value="no")
print(output)
