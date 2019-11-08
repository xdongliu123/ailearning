from .generic import GeneralLayer
from ..common.math import sigmoid, sigmoid_derivative, tanh, tanh_derivative,\
    relu, relu_derivative, softmax, softmax_derivative


class Activator(GeneralLayer):
    def __init__(self, activator):
        GeneralLayer.__init__(self)
        self.activator = activator

    def forward_propagation(self, X):
        self.X = X
        if self.activator == "sigmoid":
            self.A = sigmoid(X)
        elif self.activator == "tanh":
            self.A = tanh(X)
        elif self.activator == "relu":
            self.A = relu(X)
        elif self.activator == "softmax":
            self.A = softmax(X)
        else:
            pass
        return self.A

    def back_propagation(self, dA):
        if self.activator == "sigmoid":
            dX = sigmoid_derivative(self.X) * dA
        elif self.activator == "tanh":
            dX = tanh_derivative(self.X) * dA
        elif self.activator == "relu":
            dX = relu_derivative(self.X) * dA
        elif self.activator == "softmax":
            dX = softmax_derivative(self.X, dA)
        else:
            pass
        return dX
