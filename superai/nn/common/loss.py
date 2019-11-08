import numpy as np
from ..common.math import softmax


# y is one hot
def accuracy(A, y):
    _, n = A.shape
    # A's ele is 0 or 1
    A = (A == np.max(A, axis=0, keepdims=True))
    error = np.sum(np.abs(A - y)) / 2
    return 1 - (error / n)


def softmaxloss(X, y):
    m = y.shape[1]
    p = softmax(X)
    idx = np.squeeze(np.argmax(y, axis=0))
    log_likelihood = -np.log(p[idx, range(m)])
    loss = np.sum(log_likelihood) / m

    dx = p.copy()
    dx[idx, range(m)] -= 1
    dx /= m
    return loss, dx


def loss(A, y, cost_type):
    m = A.shape[-1]
    total_cost = 0
    eps = np.finfo(float).eps
    if cost_type == 'cross-entropy':
        total_cost = -np.sum(y * np.log(A + eps))
    elif cost_type == 'rss':
        total_cost = np.dot((A - y), (A - y).T) / m
    elif cost_type == 'softmax-loss':
        n_sum = np.sum(np.log(A) * y, axis=0, keepdims=False)
        total_cost = np.sum(n_sum) / m
    else:
        pass
    return total_cost


# logistic regression, binary class, sigmod activator
def grad_for_logistic_loss(A, y):
    # eps = np.finfo(float).eps
    g = - (np.divide(y, A) - np.divide(1 - y, (1 - A)))
    return g


def grad_for_cross_entropy_loss(A, y):
    return A - y
