import numpy as np
from .optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

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
            self.velocitys[i][j] = self.velocitys[i][j] + origin_grads[j] ** 2

    def grads_for_optimizer(self, i, model):
        adagrad_grads = []
        for j in range(len(self.velocitys[i])):
            v = self.velocitys[i][j]
            grad = model.layers[i].grads[j]
            adagrad_grads.append(grad / (np.sqrt(v) + self.epsilon))
        return adagrad_grads
