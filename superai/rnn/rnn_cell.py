import numpy as np
from superai.common.math import tanh


class RNN_CELL:
    # n_x stand for length of input vector x
    # n_a stand for length of hidden state vector a
    # n_y stand for length of output vector y
    def __init__(self, n_a, n_x, n_y):
        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.initialize_parameters()

    def initialize_parameters(self):
        self.b_a = np.random.randn(self.n_a, 1)
        self.W_a = np.random.randn(self.n_a, (self.n_a + self.n_x))
        self.b_y = np.random.randn(self.n_y, 1)
        self.W_y = np.random.randn(self.n_y, self.n_a)

    # xt's shape:(n_x, 1)
    def forward_propagation(self, xt, a):
        ax = np.append(a, xt, axis=0)
        a = np.dot(self.W_a, ax) + self.b_a
        a = tanh(a)
        z = np.dot(self.W_y, a) + self.b_y
        return z, a

    def back_propagation(self, xt, dstep, intermediates, derivatives):
        dyt, da_next = dstep
        a, a_pre = intermediates
        db_a, dW_a, db_y, dW_y = derivatives
        # calcute the derivatives
        dW_y += np.dot(dyt, a.T)
        db_y += dyt
        da = np.dot(self.W_y.T, dyt) + da_next
        # compute dtanh
        da = (1 - np.power(a, 2)) * da
        db_a += da
        axt = np.append(a_pre.T, xt.T, axis=1)
        dW_a += np.dot(da, axt)
        n_a = a.shape[0]
        da_pre = np.dot(self.W_a[:, :n_a].T, da)
        return (da_pre, )

    def update_parameters(self, derivatives, lr):
        db_a, dW_a, db_y, dW_y = derivatives
        self.b_a -= lr * db_a
        self.W_a -= lr * dW_a
        self.b_y -= lr * db_y
        self.W_y -= lr * dW_y
