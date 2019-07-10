from ..common.Utils import generate_batchs, cost_for_onehot, grad_for_onehot
import numpy as np


class CNNModel:
    def __init__(self, cnn_pipe, fc_pipe, cost_type='cross-entropy'):
        self.cnn_pipe = cnn_pipe
        self.fc_pipe = fc_pipe
        self.cost_type = cost_type

    def predict(self, X):
        return self.forward_propagation(X)

    def fit(self, X, Y, learning_rate=1.2, iteration_count=1000, lambd=0,
            use_mini_batch=False, mini_batch_size=64, print_cost=True,
            nn_optimizer="gd", nn_beta=0.9, nn_beta1=0.9, nn_beta2=0.999,
            nn_epsilon=1e-8):
        self.fc_pipe.initialize_parameters(nn_optimizer, nn_beta, nn_beta1,
                                           nn_beta2, nn_epsilon)
        batches = generate_batchs(X, Y, use_mini_batch, mini_batch_size, "cnn")
        t = 0
        for i in range(iteration_count):
            for batch in batches:
                t = t + 1
                batch_X, batch_Y = batch
                cost = (self.batch_gradient_descent_step(batch_X, batch_Y,
                        lambd, learning_rate, t))
                if print_cost:
                    print("iteration{0}    cost:{1}".format(t, cost))

    def batch_gradient_descent_step(self, X, Y, lambd, learning_rate, t):
        A = self.forward_propagation(X)
        n = X.shape[-1]
        A1 = A == np.max(A, axis=0, keepdims=True)
        C = np.abs(A1 - Y)
        error = np.sum(C).astype(np.float32) / 2
        if 1 - (error / n) < 0:
            print(C.transpose(1, 0))
        print(1 - (error / n))
        # compute cost
        cost = cost_for_onehot(A, Y, self.cost_type)
        # l2 regularization
        # pass
        self.back_propagation(A, Y, lambd, t)
        self.update_parameters(learning_rate)
        return cost

    def forward_propagation(self, X):
        m = X.shape[-1]
        cnn_output = self.cnn_pipe.forward_propagation(X)
        self.cnn_output_shape = cnn_output.shape
        # flatten multi dim to 2 keepdims
        nn_input = cnn_output.reshape(-1, m)
        A = self.fc_pipe.forward_propagation(nn_input)
        return A

    def back_propagation(self, A, Y, lambd, t):
        grad = grad_for_onehot(A, Y)
        grad_for_cnn = self.fc_pipe.back_propagation(grad, lambd, t)
        grad_for_cnn = grad_for_cnn.reshape(*(self.cnn_output_shape))
        self.cnn_pipe.back_propagation(grad_for_cnn, lambd, t)

    def update_parameters(self, learning_rate):
        self.fc_pipe.update_parameters(learning_rate)
        self.cnn_pipe.update_parameters(learning_rate)
