import numpy as np


def polynomial_expand(X, d):
    if d < 2:
        return X
    Y = X.copy()
    for i in range(2, d+1):
        X = np.row_stack((X, Y**i))
    return X


class Linear_Regression:
    def __init__(self, d):
        # 初始化参数，截距项b初始化为0
        self.w = np.random.rand(1, d+1)
        self.w[0, d] = 0

    def predict(self, X):
        m = X.shape[-1]
        # 扩展特征矩阵，将截距项stack进来
        X = np.row_stack((X, np.ones(m)))
        return self.forward_propagation(X)

    def fit(self, X, y, alpha, iteration_count=1000, verbose=True):
        m = X.shape[-1]
        X = np.row_stack((X, np.ones(m)))
        for i in range(iteration_count):
            y_pre = self.forward_propagation(X)
            loss, grad = self.back_propagation(X, y, y_pre)
            if verbose:
                print("iteration{0}  loss:{1}".format(i, loss))
            self.w = self.w - alpha * grad

    def forward_propagation(self, X):
        return np.dot(self.w, X)

    def back_propagation(self, X, y, y_pre):
        m = X.shape[-1]
        delta = y_pre - y
        grad = np.sum(delta * X, axis=1) / m
        loss = np.sum(delta ** 2) / m
        return loss, grad
