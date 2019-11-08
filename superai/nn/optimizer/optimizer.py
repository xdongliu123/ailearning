from ..common.utils import generate_batchs
from ..common.loss import softmaxloss, softmax, accuracy
import numpy as np


class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def run(self, model, X, y):
        self.init_parameters(model)
        config = model.config
        t = 0
        iteration_count = config["iteration_count"]
        for i in range(iteration_count):
            if config["use_mini_batch"]:
                batches = generate_batchs(X, y, config["mini_batch_size"])
            else:
                batches = [(X, y)]
            # iteration for adam gd
            for batch in batches:
                t = t + 1
                batch_X, batch_y = batch
                Z = model.forward_propagation(batch_X)
                A = softmax(Z)
                loss, grad = softmaxloss(Z, batch_y)
                if config["verbose"]:
                    print("iteration{0}  cost:{1}, accuracy:{2}".format(t, loss, accuracy(A, batch_y)))
                # l2 regularization
                self.back_propagation(model, grad, t)
                self.update_parameters(model)

    def init_parameters(self, model):
        self.optimizer_parameters = []
        for hidden in model.layers:
            parameters = []
            for param in hidden.params:
                v_p = np.zeros(param.shape)
                v_p_corrected = np.zeros(param.shape)
                s_p = np.zeros(param.shape)
                s_p_corrected = np.zeros(param.shape)
                parameters.append([v_p, v_p_corrected, s_p, s_p_corrected])
            self.optimizer_parameters.append(parameters)

    def back_propagation(self, model, grad, t):
        model.back_propagation(grad)
        # update adam derivations
        layer_count = len(self.optimizer_parameters)
        for i in range(layer_count):
            parameters = self.optimizer_parameters[i]
            model_parameters = model.layers[i].grads
            param_count = len(model_parameters)
            for j in range(param_count):
                opt_ps = parameters[j]
                org_p = model_parameters[j]
                v_p = self.beta1 * opt_ps[0] + (1. - self.beta1) * org_p
                v_p_corrected = v_p / (1. - self.beta1**(t))
                s_p = self.beta2 * opt_ps[2] + (1. - self.beta2) * org_p**(2)
                s_p_corrected = s_p / (1. - self.beta2**(t))
                self.optimizer_parameters[i][j] = [v_p, v_p_corrected, s_p, s_p_corrected]

    def update_parameters(self, model):
        layer_count = len(self.optimizer_parameters)
        for i in range(layer_count):
            model_layer = model.layers[i]
            param_count = len(model_layer.params)
            for j in range(param_count):
                opt_ps = self.optimizer_parameters[i][j]
                v_p_corrected = opt_ps[1]
                s_p_corrected = opt_ps[3]
                model_layer.params[j] = model_layer.params[j] - self.lr * v_p_corrected / (np.sqrt(s_p_corrected) + self.epsilon)
