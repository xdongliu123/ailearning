import numpy as np
from .linear_regression import Linear_Regression


class Logistic_Regression(Linear_Regression):
    def predict(self, X):
        m = X.shape[-1]
        X = np.row_stack((X, np.ones(m)))
        Z = self.forward_propagation(X)
        return Z >= 0.5

    def forward_propagation(self, X):
        Z = super(Logistic_Regression, self).forward_propagation(X)
        return 1 / (1 + np.exp(-Z))

    def back_propagation(self, X, y, y_pre):
        m = X.shape[-1]
        delta = y_pre - y
        grad = np.sum(delta * X, axis=1) / m
        loss = -(np.sum(y * np.log(y_pre) + (1 - y) * np.log(1 - y_pre)))
        return loss, grad
