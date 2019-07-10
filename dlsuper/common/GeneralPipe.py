class GeneralPipe:
    def __init__(self, first_layer, last_layer, *kargs, **kwargs):
        self.first_layer = first_layer
        self.last_layer = last_layer

    def forward_propagation(self, X, *kargs, **kwargs):
        hidden = self.first_layer
        while hidden is not None:
            X = hidden.forward_propagation(X, *kargs, **kwargs)
            hidden = hidden.post_layer
        return X

    def back_propagation(self, dA, *kargs, **kwargs):
        hidden = self.last_layer
        while hidden is not None:
            dA = hidden.back_propagation(dA, *kargs, **kwargs)
            hidden = hidden.pre_layer
        return dA

    def update_parameters(self, learning_rate):
        hidden = self.last_layer
        while hidden is not None:
            hidden.update_parameters(learning_rate)
            hidden = hidden.pre_layer
