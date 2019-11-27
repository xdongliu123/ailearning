import numpy as np
from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, beta=0.999, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon

    def init_optimizer_velocitys(self, model):
        self.velocitys = []
        for hidden in model.layers:
            v = [np.zeros(param.shape) for param in hidden.params]
            self.velocitys.append(v)

    def learning_rate(self, model, t):
        lr = model.config["learning_rate"]
        return lr

    def compute_velocitys(self, i, model, t):
        origin_grads = model.layers[i].grads
        for j in range(len(origin_grads)):
            v = self.velocitys[i][j]
            grad = origin_grads[j]
            self.velocitys[i][j] = self.beta * v + (1 - self.beta) * np.power(grad, 2)

    def grads_for_optimizer(self, i, model):
        rmsprop_grads = []
        for j in range(len(self.velocitys[i])):
            s = self.velocitys[i][j]
            grad = model.layers[i].grads[j]
            rmsprop_grads.append(grad / (np.sqrt(s) + self.epsilon))
        return rmsprop_grads
