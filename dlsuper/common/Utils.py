import math
import numpy as np


# activator functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    t = tanh(x)
    dt = 1 - np.power(t, 2)
    return dt


def relu(x):
    r = np.maximum(0, x)
    return r


def relu_derivative(x):
    dr = np.ones(x.shape)
    dr[x <= 0] = 0
    return dr


def cost_for_onehot(A, Y, cost_type):
    m = A.shape[-1]
    total_cost = 0
    if cost_type == 'cross-entropy':
        loss = np.multiply(np.log(A), Y) + \
               np.multiply((1 - Y), np.log(1 - A))
        total_cost = -np.sum(loss) / m
    elif cost_type == 'rss':
        total_cost = np.dot((A - Y), (A - Y).T) / m
    elif cost_type == 'softmax-loss':
        n_sum = np.sum(np.log(A) * Y, axis=0, keepdims=False)
        total_cost = np.sum(n_sum) / m
    else:
        pass
    return total_cost


def grad_for_onehot(A, Y):
    g = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    return g


def generate_batchs(X, Y, use_mini_batch, mini_batch_size, mode_type="nn"):
    batches = []
    if use_mini_batch:
        m = X.shape[-1]
        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            if mode_type == "cnn":
                mini_batch_X = X[:, :, :, k * mini_batch_size: (k + 1)
                                 * mini_batch_size]
                mini_batch_Y = Y[:, k * mini_batch_size: (k + 1)
                                 * mini_batch_size]
            else:
                mini_batch_X = X[:, k * mini_batch_size: (k + 1)
                                 * mini_batch_size]
                mini_batch_Y = Y[:, k * mini_batch_size: (k + 1)
                                 * mini_batch_size]
            batches.append((mini_batch_X, mini_batch_Y))
        if m % mini_batch_size != 0:
            if mode_type == "cnn":
                mini_batch_X = X[:, :, :, num_complete_minibatches *
                                 mini_batch_size: m]
                mini_batch_Y = Y[:, num_complete_minibatches *
                                 mini_batch_size: m]
            else:
                mini_batch_X = X[:, num_complete_minibatches *
                                 mini_batch_size: m]
                mini_batch_Y = Y[:, num_complete_minibatches *
                                 mini_batch_size: m]
            batches.append((mini_batch_X, mini_batch_Y))
    else:
        batches.append((X, Y))
    return batches
