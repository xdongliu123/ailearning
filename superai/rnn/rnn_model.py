import numpy as np
from .rnn_cell import RNN_CELL
from .lstm_cell import LSTM_CELL
from superai.common.math import sigmoid, softmax
from superai.common.loss import softmaxloss, logistic_loss


class RNN_MODEL:
    # n_x stand for length of input vector x
    # n_a stand for length of hidden state vector a
    # n_y stand for length of output vector y
    # g: softmax or sigmoid
    # cell_type: normal or lstm
    def __init__(self, n_x, n_y, n_a=100, cell_type="normal", clip_max=5, g="softmax"):
        self.g = g
        self.clip_max = clip_max
        self.cell_type = cell_type
        if cell_type == "lstm":
            self.cell = LSTM_CELL(n_a, n_x, n_y)
        else:
            self.cell = RNN_CELL(n_a, n_x, n_y)

    # x's shape:(n_x, Tx), Tx stands for time seq
    # y's shape:(n_y, Tx)
    # e.g. x = I love you, Tx = 3, n_x = total words number of Dictinary
    def train(self, x, y, learning_rate=0.01):
        Tx = x.shape[-1]
        z = np.zeros_like(y)
        if self.cell_type == "lstm":
            state = np.zeros((self.cell.n_a, 1)), np.zeros((self.cell.n_a, 1))
            states = [state]
            tmpvars = []
        else:
            state = np.zeros((self.cell.n_a, 1))
            states = [state]
        # forward propagation
        for t in range(Tx):
            ret = self.cell.forward_propagation(x[:, t:t+1], state)
            if self.cell_type == "lstm":
                z[:, t:t+1], state, vars = ret
                tmpvars.append(vars)
            else:
                z[:, t:t+1], state = ret[0], ret[1]
            states.append(state)
        # compute loss and derivative
        if self.g == "sigmoid":
            loss, dy = logistic_loss(z, y)
        else:
            loss, dy = softmaxloss(z, y)
        # back propagation
        derivatives = self._initial_derivatives()
        if self.cell_type == "lstm":
            dstate = np.zeros((self.cell.n_a, 1)), np.zeros((self.cell.n_a, 1))
        else:
            dstate = (np.zeros_like(state), )
        for t in reversed(range(Tx)):
            xt, dyt = x[:, t:t+1], dy[:, t:t+1]
            dstep = (dyt, ) + dstate
            if self.cell_type == "lstm":
                intermediates = states[t+1] + states[t] + tmpvars[t]
            else:
                intermediates = states[t+1], states[t]
            dstate = self.cell.back_propagation(xt, dstep, intermediates, derivatives)
        # clip gradients
        for gradient in derivatives:
            np.clip(gradient, -self.clip_max, self.clip_max, out=gradient)
        # update parameters
        self.cell.update_parameters(derivatives, learning_rate)
        # return loss
        return loss

    # x's shape:(n_x, Tx)
    def predict(self, x):
        Tx = x.shape[-1]
        z = np.zeros((self.cell.n_y, Tx))
        # forward propagation
        if self.cell_type == "lstm":
            state = np.zeros((self.cell.n_a, 1)), np.zeros((self.cell.n_a, 1))
        else:
            state = np.zeros((self.cell.n_a, 1))
        for t in range(Tx):
            xt = x[:, t:t+1]
            ret = self.cell.forward_propagation(xt, state)
            z[:, t:t+1], state = ret[0], ret[1]
        if self.g == "sigmoid":
            y_pre = sigmoid(z)
        else:
            y_pre = softmax(z)
        return y_pre

    def sample(self, end_index, seed):
        x = np.zeros((self.cell.n_x, 1))
        if self.cell_type == "lstm":
            state = np.zeros((self.cell.n_a, 1)), np.zeros((self.cell.n_a, 1))
        else:
            state = np.zeros((self.cell.n_a, 1))
        idx = -1
        t = 0
        idxs = []
        while idx != end_index:
            ret = self.cell.forward_propagation(x, state)
            z, state = ret[0], ret[1]
            y_pre = softmax(z)
            np.random.seed(t+seed)
            idx = np.random.choice(list(range(self.cell.n_y)), p=y_pre.ravel())
            x = np.zeros((self.cell.n_x, 1))
            x[idx, 0] = 1
            idxs.append(idx)
            seed += 1
            t += 1
        return idxs

    # private methods
    def _initial_derivatives(self):
        if self.cell_type == "lstm":
            db_f = np.zeros_like(self.cell.b_f)
            dW_f = np.zeros_like(self.cell.W_f)
            db_i = np.zeros_like(self.cell.b_i)
            dW_i = np.zeros_like(self.cell.W_i)
            db_c = np.zeros_like(self.cell.b_c)
            dW_c = np.zeros_like(self.cell.W_c)
            db_o = np.zeros_like(self.cell.b_o)
            dW_o = np.zeros_like(self.cell.W_o)
            db_y = np.zeros_like(self.cell.b_y)
            dW_y = np.zeros_like(self.cell.W_y)
            return db_f, dW_f, db_i, dW_i, db_c, dW_c, db_o, dW_o, db_y, dW_y
        else:
            db_a = np.zeros_like(self.cell.b_a)
            dW_a = np.zeros_like(self.cell.W_a)
            db_y = np.zeros_like(self.cell.b_y)
            dW_y = np.zeros_like(self.cell.W_y)
            return db_a, dW_a, db_y, dW_y
