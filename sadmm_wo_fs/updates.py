from cvxpy import *
import numpy as np
from numpy import linalg as LA
import pandas as pd


def get_data(dataset_path):
    df = pd.read_csv(dataset_path, index_col=False, dtype='float64')
    y_idx = str(df.shape[1] - 1)
    X_int = df.drop(columns=[y_idx])
    y = np.array(df[y_idx])
    X = np.array(X_int)
    X = np.c_[X, np.ones(len(y))]
    return pd.DataFrame(np.c_[X, y])


def eval_gradient(a_prev, x_i, y_i, gamma):
    if y_i * np.dot(a_prev, x_i) >= 1:
        grad = gamma * a_prev
    else:
        grad = gamma * a_prev - (y_i * x_i).T

    return grad


def eval_gradient_sadmm(a_prev, x_i, y_i):
    if y_i * np.dot(a_prev, x_i.T) >= 1:
        grad = 0
    else:
        grad = (-1) * (y_i * x_i).T

    return grad


def stochastic_x_update(data):
    node_id = data[0]
    rho = data[1]
    dataset_path = data[3]
    neighbour_data = data[4]
    a_prev = np.asmatrix(data[5]).T
    mu = data[6]

    df = get_data(dataset_path)
    dim_X = df.shape

    sample_size = 1
    sample = df.sample(sample_size, replace=True)
    y_idx = sample.shape[1] - 1
    X = np.array(sample.drop(columns=[y_idx]))
    y = np.array(sample[y_idx]).reshape((sample_size, 1))

    n = dim_X[1] - 1
    a = Variable((n, 1))
    gamma = 0.1
    grad = eval_gradient_sadmm(a_prev.T, X, y)
    g = sum(multiply(np.asmatrix(grad), a)) + gamma * (sum(multiply(a_prev, a))) + ((square(norm(a - a_prev))) / (2 * mu))
    f = 0
    for id in range(int(len(neighbour_data) / 2)):
        z = neighbour_data[id * 2].reshape((n, 1))
        u = neighbour_data[id * 2 + 1].reshape((n, 1))
        f = f + rho / 2 * square(norm(a - z + u))
    objective = Minimize(50 * g + 50 * f)
    p = Problem(objective)
    try:
        result = p.solve()
        if result is None:
            objective = Minimize(50 * g + 51 * f)
            p = Problem(objective)
            result = p.solve(verbose=False)
            if result is None:
                print("SCALING BUG")  # CVXOPT scaling issue (rarely occurs)
                objective = Minimize(52 * g + 50 * f)
                p = Problem(objective)
                p.solve(verbose=False)
    except SolverError as e:
        try:
            objective = Minimize(50 * g + 51 * f)
            p = Problem(objective)
            result = p.solve(verbose=False)
            if result is None:
                print("SCALING BUG")  # CVXOPT scaling issue (rarely occurs)
                objective = Minimize(52 * g + 50 * f)
                p = Problem(objective)
                p.solve(verbose=False)
        except:
            result = p.solve(solver=CVXOPT)
            if result is None:
                objective = Minimize(50 * g + 51 * f)
                p = Problem(objective)
                result = p.solve(verbose=False)
                if result is None:
                    print("SCALING BUG")  # CVXOPT scaling issue (rarely occurs)
                    objective = Minimize(52 * g + 50 * f)
                    p = Problem(objective)
                    p.solve(verbose=False)

    return a.value


def z_update(data):
    lamb = data[0]
    rho = data[1]
    x1 = data[2]
    x2 = data[3]
    u1 = data[4]
    u2 = data[5]
    weight = data[6]

    a = x1 + u1
    b = x2 + u2
    theta = np.maximum(1 - lamb * weight / (rho * LA.norm(a - b) + 0.000001), 0.5)  # So no divide by zero error
    z1 = theta * a + (1 - theta) * b
    z2 = theta * b + (1 - theta) * a

    return z1, z2


def u_update(data):
    x1 = data[0]
    x2 = data[1]
    z1 = data[2]
    z2 = data[3]
    u1 = data[4]
    u2 = data[5]
    u1_1 = u1 + (x1 - z1)
    u2_1 = u2 + (x2 - z2)
    return u1_1, u2_1


def rho_update(old_rho, r, s):
    mu = 10
    nu = 2
    if r > (mu * s):
        rho = nu * old_rho
    elif s > (mu * r):
        rho = old_rho / nu
    else:
        rho = old_rho
    return rho
