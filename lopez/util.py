import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def select_params(X, y, rbfk=True, C=0, gamma=0, max_accuracy=0):
    n = 5
    X_scaled = scale(X)

    print("Running RBF rounds..")
    for i in range(-5, 15, 1):
        for j in range(-15, 3, 1):
            accuracy = x_val(n, X_scaled, y, 2 ** i, True, 2 ** j)
            if accuracy >= max_accuracy:
                C = i
                gamma = j
                max_accuracy = accuracy
            else:
                break

    print("Running linear rounds..")
    prev_acc = 0
    for i in range(-5, 15):
        accuracy = x_val(n, X, y, 2 ** i, False, 0)
        if prev_acc > accuracy:
            break
        prev_acc = accuracy
        if accuracy >= max_accuracy:
            C = i
            gamma = 0
            rbfk = False
            max_accuracy = accuracy

    temp_C = C
    temp_gamma = gamma

    i = temp_C - 1
    j = temp_gamma - 1
    while i <= temp_C + 1:
        while j <= temp_gamma + 1:
            accuracy = x_val(n, X_scaled, y, 2 ** i, rbfk, 2 ** j)
            # accuracy = x_val(n, X_scaled, y, 2 ** i, rbfk, 2 ** j)
            if accuracy >= max_accuracy:
                C = i
                gamma = j
                max_accuracy = accuracy
            j += 0.25
        i += 0.25

    return 2 ** C, 2 ** gamma


def select_params_without_feature_scaling(X, y, rbfk=True, C=0, gamma=0, max_accuracy=0):
    n = 5

    for i in range(-5, 15):
        for j in range(-15, 3):
            accuracy = x_val(n, X, y, 2 ** i, True, 2 ** j)
            if accuracy >= max_accuracy:
                C = i
                gamma = j
                max_accuracy = accuracy

    for i in range(-5, 15):
        accuracy = x_val(n, X, y, 2 ** i, False, 0)
        if accuracy >= max_accuracy:
            C = i
            gamma = 0
            rbfk = False
            max_accuracy = accuracy

    temp_C = C
    temp_gamma = gamma

    i = temp_C - 1
    j = temp_gamma - 1
    while i <= temp_C + 1:
        while j <= temp_gamma + 1:
            accuracy = x_val(n, X, y, 2 ** i, rbfk, 2 ** j)
            if accuracy >= max_accuracy:
                C = i
                gamma = j
                max_accuracy = accuracy
            j += 0, 25
        i += 0.25

    return 2 ** C, 2 ** gamma


def scale(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X))


def x_val(n, X, y, C, rbfk, gamma):
    cv = KFold(n_splits=n, random_state=1, shuffle=True)
    if rbfk:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
    else:
        model = SVC(kernel="linear", C=C)

    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    return np.mean(scores)

