import abc
from ..common.utils import generate_batchs


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def init_optimizer_velocitys(self, model):
        '''please implement in subclass'''

    @abc.abstractmethod
    def compute_velocitys(self, i, model, t):
        '''please implement in subclass'''

    @abc.abstractmethod
    def grads_for_optimizer(self, i, model):
        '''please implement in subclass'''

    def learning_rate(self, model, t):
        return model.config["learning_rate"]

    def back_propagation(self, model, grad, t):
        # common back propagation
        model.back_propagation(grad)
        # update derivations with optimizer
        for i in range(len(model.layers)):
            self.compute_velocitys(i, model, t)

    def update_parameters(self, model, t):
        # print("update_parameters: %x" % id(model))
        # fetch learning rate
        lr = self.learning_rate(model, t)
        for i in range(len(model.layers)):
            grads = self.grads_for_optimizer(i, model)
            # don't update model layer's parameter of list type from outer, it is tricky
            model.layers[i].update_parameters(grads, lr)

    def run(self, model, X, y):
        self.init_optimizer_velocitys(model)
        config = model.config
        iteration_count = config["iteration_count"]
        t = 0
        for i in range(iteration_count):
            if config["use_mini_batch"]:
                batches = generate_batchs(X, y, config["mini_batch_size"])
            else:
                batches = [(X, y)]
            # iteration for adam gd
            for batch in batches:
                t = t + 1
                batch_X, batch_y = batch
                # compute current step's loss, grad and final output activator
                A, loss, grad = model.gradient_compute(batch_X, batch_y)
                # back propagation
                self.back_propagation(model, grad, t)
                self.update_parameters(model, t)
                if config["verbose"]:
                    print("iteration{0}  cost:{1}, accuracy:{2}".format(t, loss, accuracy(A, batch_y)))
