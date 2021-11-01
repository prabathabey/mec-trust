import pandas as pd
import numpy as np


def get_value(data, node_id, neighbour_id):
    try:
        neighbour_data = data[(node_id, neighbour_id)]
        val = neighbour_data[0]
    except KeyError:
        neighbour_data = data[(neighbour_id, node_id)]
        val = neighbour_data[1]

    return val


def calculate_accuracy(x, iter, num_nodes, base_dataset_path):
    total_correct_preds = 0
    num_total_test_samples = 0

    for node_id in range(num_nodes):
        test_dataset_path = base_dataset_path + "/test-" + str(node_id) + ".csv"
        X_test, y_test = get_data(test_dataset_path)

        num_test_samples = y_test.shape[0]
        num_total_test_samples += num_test_samples
        a = np.array(x[:, node_id])
        a = a.reshape(1, a.shape[0])
        y_pred = np.sign(np.dot(X_test, a.T)).flatten()

        correct_preds = int(np.sum(np.abs(y_pred - y_test)) / 2)
        total_correct_preds += correct_preds

    if iter is None:
        print("Global Accuracy: " + str(total_correct_preds / float(num_total_test_samples)))
    else:
        print("Iteration: " + str(iter) + ", Accuracy: " + str(total_correct_preds / float(num_total_test_samples)))


def get_data(dataset_path):
    rawData = pd.read_csv(dataset_path, index_col=False, dtype='float64')
    X = np.array(rawData.drop(columns=[str(rawData.shape[1] - 1)]))
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    y = np.array(rawData[str(rawData.shape[1] - 1)])
    return X, y
