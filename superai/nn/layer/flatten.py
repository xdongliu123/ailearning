from .generic import GeneralLayer


class Flatten(GeneralLayer):
    def __init__(self):
        GeneralLayer.__init__(self)

    def forward_propagation(self, X):
        self.input_shape = X.shape
        m = self.input_shape[-1]
        # flatten multi dim to 2 keepdims
        output = X.reshape(-1, m)
        return output

    def back_propagation(self, dout, l1_lambd, l2_lambd):
        grad = dout.reshape(*(self.input_shape))
        return grad
