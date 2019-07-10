from .CNNPipe import CNNPipe
import numpy as np


class CNNStackNode:
    def __init__(self, *input_layers):
        self.input_layers = input_layers
        for layer in input_layers:
            assert isinstance(layer, CNNPipe)or isinstance(input, CNNStackNode)

    def forward_propagation(self, X):
        outputs = (layer.forward_propagation(X) for layer in self.input_layers)
        self.outputs = outputs
        A = np.concatenate(outputs, axis=2)
        return A

    def back_propagation(self, dA):
        start = 0
        for i in len(self.input_layers):
            input_layer = self.input_layers[i]
            last_dim_len = self.outputs[i].shape[self.outputs[i].ndim - 1]
            end = start + last_dim_len - 1
            input_layer.back_propagation(dA[:, :, start:end, :])
            start = start + last_dim_len

    def update_parameters(self, learning_rate):
        for input_layer in self.input_layers:
            input_layer.update_parameters(learning_rate)
