
import numpy as np
from .generic import GeneralLayer


class FullyConnected(GeneralLayer):
    # nx: size of input vector
    # nh: size of output vector
    def __init__(self, n_x, n_h):
        GeneralLayer.__init__(self)
        self.shape = (n_h, n_x)
        # init parameters
        np.random.seed(1)
        self.W = np.random.randn(n_x, n_h) / np.sqrt(n_x / 2.)
        self.W = self.W.transpose()
        self.b = np.zeros((n_h, 1))
        self.params.append(self.W)
        self.params.append(self.b)

    def forward_propagation(self, X):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        return self.Z

    def back_propagation(self, dZ, l1_lambd, l2_lambd):
        self.dW = np.dot(dZ, self.X.T)

        m = dZ.shape[-1]
        # consider l1 regularization
        self.dW += l1_lambd * self.W / ((np.abs(self.W) + 1e-8) * m)
        # consider l2 regularization
        self.dW += (l2_lambd * self.W) / m

        self.db = np.sum(dZ, axis=1, keepdims=True)
        self.grads = [self.dW, self.db]
        dX = np.dot(self.W.T, dZ)
        return dX

    def regularization_cost(self, l1_lambd, l2_lambd):
        return 0.5 * l2_lambd * np.sum(self.W * self.W) + l1_lambd * np.sum(np.abs(self.W))
