import numpy as np
from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, beta=0.9):
        self.beta = beta

    def init_optimizer_velocitys(self, model):
        self.velocitys = []
        for hidden in model.layers:
            v = [np.zeros(param.shape) for param in hidden.params]
            self.velocitys.append(v)

    def learning_rate(self, model, t):
        return model.config["learning_rate"]

    def compute_velocitys(self, i, model, t):
        origin_grads = model.layers[i].grads
        for j in range(len(origin_grads)):
            v = self.velocitys[i][j]
            grad = origin_grads[j]
            self.velocitys[i][j] = self.beta * v + (1 - self.beta) * grad

    def grads_for_optimizer(self, i, model):
        momentum_grads = []
        for j in range(len(self.velocitys[i])):
            momentum_grads.append(self.velocitys[i][j])
        return momentum_grads
