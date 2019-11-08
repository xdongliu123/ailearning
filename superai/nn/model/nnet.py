from ..common.utils import generate_batchs
from ..common.loss import softmaxloss, softmax, accuracy


class Model:
    pass


class Sequence:
    def __init__(self, layes=[], learning_rate=0.1, iteration_count=1000, lambd=0,
                 use_mini_batch=False, mini_batch_size=64, verbose=True, loss_type="softmax"):
        self.layers = layes
        self.config = {"iteration_count": iteration_count, "lambd": lambd,
                       "use_mini_batch": use_mini_batch, "mini_batch_size": mini_batch_size,
                       "verbose": verbose, "learning_rate": learning_rate, "loss_type": loss_type}

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
                batch_X, batch_y = batch
                self.batch_gradient_descent_step(batch_X, batch_y, t)
                t = t + 1

    def batch_gradient_descent_step(self, X, y, t):
        Z = self.forward_propagation(X)
        A = softmax(Z)
        loss, grad = softmaxloss(Z, y)
        if self.config["verbose"]:
            print("iteration{0}  cost:{1}, accuracy:{2}".format(t, loss, accuracy(A, y)))
        # l2 regularization
        self.back_propagation(grad)
        self.update_parameters(self.config["learning_rate"])

    def forward_propagation(self, X):
        for hidden in self.layers:
            X = hidden.forward_propagation(X)
        return X

    def back_propagation(self, grad):
        for hidden in reversed(self.layers):
            grad = hidden.back_propagation(grad)

    def update_parameters(self, learning_rate):
        for hidden in reversed(self.layers):
            hidden.update_parameters(hidden.grads, learning_rate)
