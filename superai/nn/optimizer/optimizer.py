from ..common.utils import generate_batchs
from ..common.loss import softmaxloss, softmax, accuracy
import numpy as np


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def run(self, model, X, y):
        self.init_adam_grads(model)
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
                self.update_parameters(model, t)

    def init_adam_grads(self, model):
        self.optimizer_grads = [(np.zeros(param.shape), np.zeros(param.shape), np.zeros(param.shape), np.zeros(param.shape)) for hidden in model.layers for param in hidden.params]

    def back_propagation(self, model, grad, t):
        model.back_propagation(grad)
        # update adam derivations
        base = 0
        for i in range(len(model.layers)):
            for j in range(len(model.layers[i].grads)):
                index = base + j
                v_p, v_p_c, s_p, s_p_c = self.optimizer_grads[index]
                grad = model.layers[i].grads[j]
                v_p = self.beta1 * v_p + (1. - self.beta1) * grad
                v_p_corrected = v_p / (1. - self.beta1**(t))
                s_p = self.beta2 * s_p + (1. - self.beta2) * grad**(2)
                s_p_corrected = s_p / (1. - self.beta2**(t))
                self.optimizer_grads[index] = (v_p, v_p_corrected, s_p, s_p_corrected)
            base = base + len(model.layers[i].grads)

    def update_parameters(self, model, t):
        # print("update_parameters: %x" % id(model))
        lr = model.config["learning_rate"]
        lr_t = lr * (np.sqrt(1. - np.power(self.beta2, t)) / (1. - np.power(self.beta1, t)))
        base = 0
        for i in range(len(model.layers)):
            adam_grads = []
            for j in range(len(model.layers[i].params)):
                index = base + j
                v_p, v_p_c, s_p, s_p_c = self.optimizer_grads[index]
                adam_grads.append(v_p_c / (np.sqrt(s_p_c) + self.epsilon))
            # don't update model layer's parameter of list type from outer, it is tricky
            model.layers[i].update_parameters(adam_grads, lr_t)
            base = base + len(model.layers[i].params)
