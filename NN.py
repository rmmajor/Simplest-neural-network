import numpy as np
import matplotlib.pyplot as plt
from test_set import initialize_set

X_train, Y_train = initialize_set(nx=3, m=2000)
X_test, Y_test = initialize_set()


# m_train = X_train.shape[0]
# nx = X_train.shape[1]
# m_test = X_test.shape[0]


def sigmoid(X):
    A = 1 / (1 + np.exp(-X))
    cache = X
    return A, cache


def sigmoid_derivative(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu(X):
    A = np.maximum(X, 0)
    cache = X
    return A, cache


def relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0  # if Z[i][j] <= 0, then dZ[i][j] = 0, else nothing changes
    return dZ


def tanh(X):
    A = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    cache = X
    return A, cache


def tanh_derivative(dA, cache):
    # add later
    pass


def leaky_relu(X):
    A = np.maximum(0.01 * X, X)
    cache = X
    return A, cache


def leaky_relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.01  # if Z[i][j] <= 0, then dZ[i][j] = 0.01, else nothing changes
    return dZ


def init_params(layer_sz):
    # np.random.seed(3)
    parameters = {}
    L = len(layer_sz)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(int(layer_sz[l]), int(layer_sz[l - 1])) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_sz[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_sz[l], layer_sz[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_sz[l], 1))
    # print(parameters)
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def activation_forward(A_prev, W, b, activation='relu'):
    Z, linear_cache = linear_forward(A_prev, W, b)
    # print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
    # print(Z)
    # print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
    activation_cache = None
    if activation == 'relu':
        A, activation_cache = relu(Z)
    if activation == 'tanh':
        A, activation_cache = tanh(Z)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    if activation == 'leaky_relu':
        A, activation_cache = leaky_relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    # print(A)
    # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    return A, cache


def L_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)

    AL, cache = activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    assert(AL.shape == (1, X.shape[1]))
    caches.append(cache)

    # print()
    # print(caches)
    # print()

    return AL, caches


def cost_func(AL, Y):
    m = AL.shape[1]
    # cost = 0
    cost = np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / (-m)
    print('Al and Y')
    # print(AL, Y)
    print('cost1')
    # print((np.multiply(Y, np.log(AL)) ))
    # print()
    # print(np.multiply(1 - Y, np.log(1 - AL)))

    cost = np.squeeze(cost)  # this turns [[17]] into 17
    assert (cost.shape == ())
    print(cost)
    return cost


def lineral_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def activation_backward(dA, cache, activation='relu'):
    linear_cache, activation_cache = cache
    dZ = None
    if activation == 'relu':
        dZ = relu_derivative(dA, activation_cache)
        # print('here')
        # dA_prev, dW, db = lineral_backward(dZ, linear_cache)

    elif activation == 'leaky_relu':
        dZ = leaky_relu_derivative(dA, activation_cache)
        # dA_prev, dW, db = lineral_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_derivative(dA, activation_cache)
        # dA_prev, dW, db = lineral_backward(dZ, linear_cache)

    elif activation == 'tanh':
        dZ = tanh_derivative(dA, activation_cache)
        # dA_prev, dW, db = lineral_backward(dZ, linear_cache)

    dA_prev, dW, db = lineral_backward(dZ, linear_cache)
    # print(dZ)
    # print(dA_prev, dW, db)
    return dA_prev, dW, db


def L_backward(AL, Y, caches):
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    grads = {}
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache,
                                                                                               activation='sigmoid')

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    # print(grads)
    return grads


def upd(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        # print(grads)
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=2500):
    # np.random.seed(1)
    costs = []
    parameters = init_params(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_forward(X, parameters)
        cost = cost_func(AL, Y)
        grads = L_backward(AL, Y, caches)
        parameters = upd(parameters, grads, learning_rate)
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    # p = np.zeros((1,m))

    probas, caches = L_forward(X, parameters)

    print(probas)

    # for i in range(0, probas.shape[1]):
    #     if probas[0,i] > 0:


layers_dims = [3, 3, 1]
parameters = L_layer_model(X_train, Y_train, layers_dims, num_iterations=2500)
