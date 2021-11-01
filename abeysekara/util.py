import pandas as pd
import numpy as np

np.random.seed(10)


def get_value(data, node_id, neighbour_id):
    try:
        neighbour_data = data[(node_id, neighbour_id)]
        val = neighbour_data[0]
    except KeyError:
        neighbour_data = data[(neighbour_id, node_id)]
        val = neighbour_data[1]

    return val


def calculate_consensus(z, edges):
    num_edges = len(z)
    cons = 0
    for edge in edges:
        z_edge = z[edge]
        if np.all(z_edge[0] == z_edge[1]):
            cons += 1
    return cons / float(num_edges)


def calculate_accuracy(x, num_nodes, base_dataset_path):
    total_correct_preds = 0
    total_test_samples = 0
    for node_id in range(num_nodes):
        test_dataset_path = base_dataset_path + "test-" + str(node_id) + ".csv"
        X_test, y_test = get_data(test_dataset_path)

        total_test_samples += y_test.shape[0]

        a = np.array(x[:, node_id]).reshape((X_test.shape[1], 1))
        y_pred = np.sign([np.dot(X_test, a)]).flatten()
        correct_preds = y_test.shape[0] - int(np.sum(np.abs(y_pred - y_test))/2)
        total_correct_preds += correct_preds

    return total_correct_preds / float(total_test_samples)


def get_data(dataset_path):
    df = pd.read_csv(dataset_path, index_col=False, dtype='float64')
    y_idx = str(df.shape[1] - 1)
    X_int = df.drop(columns=[y_idx])
    y = np.array(df[y_idx])
    X = np.array(X_int)
    X = np.c_[X, np.ones(len(y))]
    return X, y
