from ..common.GeneralPipe import GeneralPipe
from .CNNLayer import CNNLayer


class CNNPipe(GeneralPipe):
    def initialize_parameters(self, optimizer, beta, beta1, beta2, epsilon):
        hiddenLayer = self.first_layer
        while hiddenLayer is not None:
            if isinstance(hiddenLayer, CNNLayer):
                hiddenLayer.initialize_parameters(optimizer, beta, beta1,
                                                  beta2, epsilon)
            hiddenLayer = hiddenLayer.post_layer
