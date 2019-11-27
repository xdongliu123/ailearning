from .generic import GeneralLayer
import numpy as np


class NNDropout(GeneralLayer):
    def __init__(self, keep_prob):
        GeneralLayer.__init__(self)
        self.keep_prob = keep_prob

    def forward_propagation(self, X):
        random = np.random.rand(X.shape)
        self.mask = (random < self.keep_prob)
        return (X * self.mask) / self.keep_prob

    def back_propagation(self, dA, l1_lambd, l2_lambd):
        return (dA * self.mask) / self.keep_prob
