import abc


class GeneralLayer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.params = []
        self.grads = []

    @abc.abstractmethod
    def forward_propagation(self, X):
        '''please implement in subclass'''

    @abc.abstractmethod
    def back_propagation(self, dA, l1_lambd, l2_lambd):
        '''please implement in subclass'''

    def regularization_cost(self, l1_lambd, l2_lambd):
        return 0

    def update_parameters(self, grads, learning_rate):
        assert len(self.params) == len(grads), "parameter and derivative unpair"
        for i in range(len(grads)):
            assert self.params[i].shape == grads[i].shape, "parameter and derivative unpair"
            self.params[i] += - learning_rate * grads[i]
