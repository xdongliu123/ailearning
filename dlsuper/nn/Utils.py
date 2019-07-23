import numpy as np


def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=0, keepdims=True))
    # + 1e-11
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s


def softmax_grad_vector(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix
    # multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is
    # usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


def softmax_derivative(x, dA):
    s = softmax(x)
    m = dA.shape[1]
    ds = np.zeros(x.shape)
    for i in range(m):
        da = dA[:, i]
        sg = softmax_grad_vector(s[:, i])
        ds[:, i] = np.dot(da, sg)
    return ds
