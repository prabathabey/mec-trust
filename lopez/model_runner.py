from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from lopez.util import select_params
import numpy as np
import pandas as pd

experiments = [100, 150, 200, 250, 300, 350, 400]
base_paths = ["/datasets/unsw-nb15", "/datasets/bot-iot", "/datasets/n-baiot"]

for base_path in base_paths:
    for num_nodes in experiments:
        for node_id in range(1, num_nodes + 1):
            df = pd.read_csv(base_path + str(node_id) + ".csv", header=None)

            X = df.loc[:, df.columns != 115]
            y = df[115]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=node_id)

            C, gamma = select_params(X_train, y_train, rbfk=False)
            clf = SVC(kernel="rbf", C=C, gamma=gamma)
            clf.fit(X=X_train, y=y_train)
            y_hat = clf.predict(X_test)

            error = np.sum(np.abs(y_hat - y_test)) / len(y_test)
            accuracy = 1.0 - error

            print("Node id: " + str(node_id) + ", Accuracy:" + str(accuracy))