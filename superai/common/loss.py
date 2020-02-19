import numpy as np
from ..common.math import softmax, sigmoid


# y is one hot
def accuracy(y_pre, y):
    _, n = y_pre.shape
    # A's ele is 0 or 1
    y_pre = (y_pre == np.max(y_pre, axis=0, keepdims=True))
    error = np.sum(np.abs(y_pre - y)) / 2
    return 1 - (error / n)


# X, y is 2-dim arrar of shape(d, m), the value of y element is 0 or 1
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


# logistic regression, binary class, sigmod activator
# y should be vector of shape(1, m), the value of y element is 0 or 1
def logistic_loss(X, y):
    # eps = np.finfo(float).eps
    y_pre = sigmoid(X)
    grad = y_pre - y
    loss = -(np.sum(y * np.log(y_pre) + (1 - y) * np.log(1 - y_pre)))
    return loss, grad
