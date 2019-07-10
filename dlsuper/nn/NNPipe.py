import numpy as np
from ..common.GeneralPipe import GeneralPipe
from .NNLayer import NNLinearLayer


class NNPipe(GeneralPipe):
    def initialize_parameters(self, optimizer, beta, beta1, beta2, epsilon):
        hiddenLayer = self.first_layer
        while hiddenLayer is not None:
            if isinstance(hiddenLayer, NNLinearLayer):
                hiddenLayer.initialize_parameters(optimizer, beta, beta1,
                                                  beta2, epsilon)
            hiddenLayer = hiddenLayer.post_layer

    def l2_regularization_cost(self, m, lambd):
        L2_cost = 0
        hidden = self.last_layer
        while hidden is not None:
            if isinstance(hidden, NNLinearLayer):
                L2_cost += np.sum(np.square(hidden.W))
            hidden = hidden.pre_layer
        return L2_cost * lambd / (2 * m)
