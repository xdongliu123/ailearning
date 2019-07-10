import abc


class GeneralLayer(metaclass=abc.ABCMeta):
    def __init__(self, pre_layer, *kargs, **kwargs):
        self.pre_layer = pre_layer
        self.post_layer = None
        if pre_layer is not None:
            self.pre_layer.post_layer = self

    @abc.abstractmethod
    def forward_propagation(self, X, *kargs, **kwargs):
        '''please implement in subclass'''

    @abc.abstractmethod
    def back_propagation(self, dA, *kargs, **kwargs):
        '''please implement in subclass'''

    def update_parameters(self, learning_rate):
        pass
