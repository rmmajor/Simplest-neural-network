import numpy as np
import random


def initialize_set(nx = 3, m = 1):
    X = np.random.randint(0, 50, (nx, m))
    Y = np.zeros((m, 1))
    for i in range(m):
        Y[i] = X[0, i] ^ X[1, i] ^ X[2, i]
    return X, Y


# Z = np.random.randint(-50, 50, (10, 1))
# print(Z)
# Z_relu = np.maximum(Z, 0)
# print(Z_relu)

# print(X)
# print(Y)
