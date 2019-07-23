import numpy as np
from .NNPipe import NNPipe
from ..common.Utils import generate_batchs, cost_for_onehot, grad_for_onehot


class NNModel:
    def __init__(self, pipe, cost_type='cross-entropy', normalization=False,
                 random_seed=3):
        assert isinstance(pipe, NNPipe), "pipe type invalid"
        self.pipe = pipe
        self.normalization = normalization
        self.cost_type = cost_type
        np.random.seed(random_seed)

    def predict(self, X):
        if self.normalization:
            norm = np.linalg.norm(X, axis=0, keepdims=True)
            X = X / norm
        return self.forward_propagation(X, keep_prob=1)

    # optimizer:gd, momentum, adam
    def fit(self, X, Y, learning_rate=1.2, iteration_count=1000, lambd=0,
            keep_prob=1, use_mini_batch=False, mini_batch_size=64,
            optimizer="adam", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
            print_cost=True):
        if self.normalization:
            norm = np.linalg.norm(X, axis=0, keepdims=True)
            X = X / norm
        self.pipe.initialize_parameters(optimizer, beta, beta1, beta2, epsilon)

        # generare batches of gradient descent
        t = 0
        n = Y.shape[1]
        for i in range(iteration_count):
            if use_mini_batch:
                order = np.arange(n)
                np.random.shuffle(order)
                X_input = X[:, :, :, order]
                Y_input = Y[:, order]
            else:
                X_input = X[:, :, :, :]
                Y_input = Y[:, :]
            batches = generate_batchs(X_input, Y_input, use_mini_batch,
                                      mini_batch_size)
            for batch in batches:
                t = t + 1
                batch_X, batch_Y = batch
                cost = (self.batch_gradient_descent_step(batch_X, batch_Y,
                        keep_prob, lambd, learning_rate, t))
            if print_cost and i % 100 == 0:
                print("iteration{0}    cost:{1}".format(i, cost))

    def batch_gradient_descent_step(self, X, Y, keep_prob, lambd,
                                    learning_rate, t):
        A = self.forward_propagation(X, keep_prob)
        m = A.shape[-1]
        # compute cost
        cost = cost_for_onehot(A, Y, self.cost_type)
        if lambd != 0:
            cost += self.pipe.l2_regularization_cost(m, lambd)

        self.back_propagation(A, Y, lambd, t)
        self.update_parameters(learning_rate)
        return cost

    def forward_propagation(self, X, keep_prob):
        return self.pipe.forward_propagation(X, keep_prob)

    def back_propagation(self, A, Y, lambd, t):
        grad = grad_for_onehot(A, Y)
        self.pipe.back_propagation(grad, lambd, t)

    def update_parameters(self, learning_rate):
        self.pipe.update_parameters(learning_rate)
