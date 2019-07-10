import math
import numpy as np
from ..common.GeneralLayer import GeneralLayer
from ..common.Utils import sigmoid, sigmoid_derivative, tanh, tanh_derivative,\
    relu, relu_derivative
from .Utils import softmax, softmax_derivative


"""
Neural network layer
n_x: input vector size, n_h: output vector size
preLayer => layer  => postLayer
"""


class NNLinearLayer(GeneralLayer):
    def __init__(self, pre_layer, n_x, n_h, initialize_method="normal"):
        GeneralLayer.__init__(self, pre_layer)
        self.n_x = n_x
        self.n_h = n_h
        self.shape = (n_h, n_x)
        self.initialize_method = initialize_method
        if pre_layer is not None:
            assert(self.shape[1] == pre_layer.shape[0])

    def initialize_parameters(self, optimizer, beta, beta1, beta2, epsilon):
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        if self.initialize_method == 'He':
            self.W = np.random.randn(self.shape[0], self.shape[1]) \
             * math.sqrt(2 / self.n_x)
            self.b = np.zeros((self.n_h, 1))
        elif self.initialize_method == 'normal':
            self.W = np.random.randn(self.shape[0], self.shape[1])
            self.b = np.zeros((self.n_h, 1))
        else:
            self.W = np.random.randn(self.shape[0], self.shape[1])
            self.b = np.zeros((self.n_h, 1))

        '''
        Special infrastructure for optimizers:[Momentum, RMSprop, Adam]
        '''
        if self.optimizer == "momentum":
            self.v_dW = np.zeros(self.W.shape)
            self.v_db = np.zeros(self.b.shape)
        elif self.optimizer == "adam":
            self.v_dW = np.zeros(self.W.shape)
            self.v_db = np.zeros(self.b.shape)
            self.s_dW = np.zeros(self.W.shape)
            self.s_db = np.zeros(self.b.shape)
        else:
            pass

    def forward_propagation(self, X, *kargs, **kwargs):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        return self.Z

    def back_propagation(self, dZ, *kargs, **kwargs):
        lambd = kargs[0]
        t = kargs[1]
        m = dZ.shape[1]
        self.dW = np.dot(dZ, self.X.T) / m + lambd * self.W / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m

        if self.optimizer == "momentum":
            self.v_dW = self.beta * self.v_dW + (1 - self.beta) * self.dW
            self.v_db = self.beta * self.v_db + (1 - self.beta) * self.db
        elif self.optimizer == "adam":
            self.v_dW = self.beta1 * self.v_dW + (1 - self.beta1) * self.dW
            self.v_db = self.beta1 * self.v_db + (1 - self.beta1) * self.db
            self.v_dW_corrected = self.v_dW / (1 - np.power(self.beta1, t))
            self.v_db_corrected = self.v_db / (1 - np.power(self.beta1, t))

            self.s_dW = self.beta2 * self.s_dW + (1 - self.beta2) * \
                np.power(self.dW, 2)
            self.s_db = self.beta2 * self.s_db + (1 - self.beta2) * \
                np.power(self.db, 2)
            self.s_dW_corrected = self.s_dW / (1 - np.power(self.beta2, t))
            self.s_db_corrected = self.s_db / (1 - np.power(self.beta2, t))
        else:
            pass
        dX = np.dot(self.W.T, dZ)
        return dX

    def update_parameters(self, learning_rate):
        if self.optimizer == "gd":
            self.W = self.W - learning_rate * self.dW
            self.b = self.b - learning_rate * self.db
        elif self.optimizer == "momentum":
            self.W = self.W - learning_rate * self.v_dW
            self.b = self.b - learning_rate * self.v_db
        elif self.optimizer == "adam":
            self.W = self.W - learning_rate * self.v_dW_corrected / \
                (np.sqrt(self.s_dW_corrected) + self.epsilon)
            self.b = self.b - learning_rate * self.v_db_corrected / \
                (np.sqrt(self.s_db_corrected) + self.epsilon)


class NNActivator(GeneralLayer):
    def __init__(self, pre_layer, activator, n_x):
        GeneralLayer.__init__(self, pre_layer)
        self.n_x = n_x
        self.n_h = n_x
        self.shape = (n_x, n_x)
        self.activator = activator

    def forward_propagation(self, X, *kargs, **kwargs):
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

    def back_propagation(self, dA, *kargs, **kwargs):
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


class NNDropout(GeneralLayer):
    def __init__(self, pre_layer, n_x,):
        GeneralLayer.__init__(self, pre_layer)
        self.n_x = n_x
        self.n_h = n_x
        self.shape = (n_x, n_x)

    def forward_propagation(self, X, keep_prob, *kargs, **kwargs):
        self.keep_prob = keep_prob
        random = np.random.rand(X.shape)
        self.mask = (random < keep_prob)
        return (X * self.mask) / keep_prob

    def back_propagation(self, dA, *kargs, **kwargs):
        return (dA * self.mask) / self.keep_prob
