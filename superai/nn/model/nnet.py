from ..common.utils import generate_batchs
from superai.common.loss import softmaxloss, softmax, accuracy


class Sequence:
    def __init__(self, layes=[], learning_rate=0.1, iteration_count=1000, l1_lambd=0, l2_lambd=0,
                 use_mini_batch=False, mini_batch_size=64, verbose=True):
        self.layers = layes
        self.config = {"iteration_count": iteration_count, "l1_lambd": l1_lambd, "l2_lambd": l2_lambd,
                       "use_mini_batch": use_mini_batch, "mini_batch_size": mini_batch_size,
                       "verbose": verbose, "learning_rate": learning_rate}

    def predict(self, X):
        return softmax(self.forward_propagation(X))

    def fit(self, X, y):
        t = 0
        iteration_count = self.config["iteration_count"]
        for i in range(iteration_count):
            if self.config["use_mini_batch"]:
                batches = generate_batchs(X, y, self.config["mini_batch_size"])
            else:
                batches = [(X, y)]
            # iteration for gd
            for batch in batches:
                t = t + 1
                batch_X, batch_y = batch
                A, loss, grad = self.gradient_compute(batch_X, batch_y)
                self.back_propagation(grad)
                self.update_parameters(self.config["learning_rate"])
                if self.config["verbose"]:
                    print("iteration{0}  cost:{1}, accuracy:{2}".format(t, loss, accuracy(A, batch_y)))

    def gradient_compute(self, X, y):
        Z = self.forward_propagation(X)
        loss, grad = softmaxloss(Z, y)
        # add regular lost
        loss += self.regularization_cost(X.shape[-1])
        return softmax(Z), loss, grad

    def forward_propagation(self, X):
        for hidden in self.layers:
            X = hidden.forward_propagation(X)
        return X

    def back_propagation(self, grad):
        l1_lambd = self.config["l1_lambd"]
        l2_lambd = self.config["l2_lambd"]
        for hidden in reversed(self.layers):
            grad = hidden.back_propagation(grad, l1_lambd, l2_lambd)

    def update_parameters(self, learning_rate):
        for hidden in reversed(self.layers):
            hidden.update_parameters(hidden.grads, learning_rate)

    def regularization_cost(self, m):
        l1_lambd = self.config["l1_lambd"]
        l2_lambd = self.config["l2_lambd"]
        regular_lost = 0
        for hidden in reversed(self.layers):
            regular_lost += hidden.regularization_cost(l1_lambd, l2_lambd)
        return regular_lost / m
