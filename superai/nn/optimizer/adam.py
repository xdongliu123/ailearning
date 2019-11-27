import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init_optimizer_velocitys(self, model):
        self.velocitys = []
        for hidden in model.layers:
            v = [(np.zeros(param.shape), np.zeros(param.shape), np.zeros(param.shape), np.zeros(param.shape)) for param in hidden.params]
            self.velocitys.append(v)

    def learning_rate(self, model, t):
        lr = model.config["learning_rate"]
        return lr * (np.sqrt(1. - np.power(self.beta2, t)) / (1. - np.power(self.beta1, t)))

    def compute_velocitys(self, i, model, t):
        origin_grads = model.layers[i].grads
        for j in range(len(origin_grads)):
            v_p, v_p_c, s_p, s_p_c = self.velocitys[i][j]
            grad = origin_grads[j]
            v_p = self.beta1 * v_p + (1. - self.beta1) * grad
            v_p_corrected = v_p / (1. - self.beta1**(t))
            s_p = self.beta2 * s_p + (1. - self.beta2) * grad**(2)
            s_p_corrected = s_p / (1. - self.beta2**(t))
            self.velocitys[i][j] = (v_p, v_p_corrected, s_p, s_p_corrected)

    def grads_for_optimizer(self, i, model):
        adam_grads = []
        for j in range(len(self.velocitys[i])):
            v_p, v_p_c, s_p, s_p_c = self.velocitys[i][j]
            adam_grads.append(v_p_c / (np.sqrt(s_p_c) + self.epsilon))
        return adam_grads
