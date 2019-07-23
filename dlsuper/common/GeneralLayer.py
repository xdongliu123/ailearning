import abc
import numpy as np


class GeneralLayer(metaclass=abc.ABCMeta):
    def __init__(self, pre_layer, *kargs, **kwargs):
        self.pre_layer = pre_layer
        self.post_layer = None
        if pre_layer is not None:
            self.pre_layer.post_layer = self

    def initialize_parameters(self, optimizer, beta, beta1, beta2, epsilon):
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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

    @abc.abstractmethod
    def forward_propagation(self, X, *kargs, **kwargs):
        '''please implement in subclass'''

    @abc.abstractmethod
    def back_propagation(self, dA, *kargs, **kwargs):
        '''please implement in subclass'''

    def update_parameters(self, learning_rate):
        pass

    def adjust_derivation_parameters(self, t):
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
