import numpy as np
from superai.common.math import tanh, sigmoid


class LSTM_CELL:
    # n_x stand for length of input vector x
    # n_a stand for length of hidden state vector a
    # n_y stand for length of output vector y
    def __init__(self, n_a, n_x, n_y):
        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.initialize_parameters()

    def initialize_parameters(self):
        self.b_f = np.random.randn(self.n_a, 1)
        self.W_f = np.random.randn(self.n_a, (self.n_a + self.n_x))
        self.b_i = np.random.randn(self.n_a, 1)
        self.W_i = np.random.randn(self.n_a, (self.n_a + self.n_x))
        self.b_c = np.random.randn(self.n_a, 1)
        self.W_c = np.random.randn(self.n_a, (self.n_a + self.n_x))
        self.b_o = np.random.randn(self.n_a, 1)
        self.W_o = np.random.randn(self.n_a, (self.n_a + self.n_x))
        self.b_y = np.random.randn(self.n_y, 1)
        self.W_y = np.random.randn(self.n_y, self.n_a)

    # x's shape:(n_x, 1)
    # y's shape:(n_y, 1)
    def forward_propagation(self, xt, state):
        a, c = state
        ax = np.append(a, xt, axis=0)
        ft = sigmoid(np.dot(self.W_f, ax) + self.b_f)
        it = sigmoid(np.dot(self.W_i, ax) + self.b_i)
        cct = tanh(np.dot(self.W_c, ax) + self.b_c)
        c = ft * c + it * cct
        ot = sigmoid(np.dot(self.W_o, ax) + self.b_o)
        a = ot * tanh(c)
        z = np.dot(self.W_y, a) + self.b_y
        return z, (a, c), (ft, it, cct, ot)

    def back_propagation(self, xt, dstep, intermediates, derivatives):
        dyt, da_next, dc_next = dstep
        a, c, a_pre, c_pre, ft, it, cct, ot = intermediates
        db_f, dW_f, db_i, dW_i, db_c, dW_c, db_o, dW_o, db_y, dW_y = derivatives
        ax = np.append(a_pre, xt, axis=0)
        # calcute the derivatives
        dW_y += np.dot(dyt, a.T)
        db_y += dyt
        da = np.dot(self.W_y.T, dyt) + da_next
        dot = tanh(c) * da
        # dsigmoid:s * (1 - s)
        dot = (ot * (1 - ot)) * dot
        dW_o += np.dot(dot, ax.T)
        db_o += dot
        # compute dc: from two branches
        # dtanh: 1 - np.power(t, 2)
        dc = ot * da * (1 - np.power(c, 2))
        dc += dc_next
        # comput:dft,dit,dcct
        dft = c_pre * dc
        dft = (ft * (1 - ft)) * dft
        dit = cct * dc
        dit = (it * (1 - it)) * dit
        dcct = it * dc
        dcct = (1 - np.power(cct, 2)) * dcct
        # comput dW_f, dW_i, dW_c...
        dW_c += np.dot(dcct, ax.T)
        db_c += dcct
        dW_i += np.dot(dit, ax.T)
        db_i += dit
        dW_f += np.dot(dft, ax.T)
        db_f += dft
        # compute da_pre, dc_pre
        da_pre = np.dot(self.W_f[:, :self.n_a].T, dft) + np.dot(self.W_i[:, :self.n_a].T, dit)\
            + np.dot(self.W_c[:, :self.n_a].T, dcct) + np.dot(self.W_o[:, :self.n_a].T, dot)
        # dc_prev from [c = ft * c + it * cct] and [a = ot * tanh(c)]
        dc_prev = dc_next * ft + ot * (1 - np.power(tanh(c), 2)) * ft * da_next
        return da_pre, dc_prev

    def update_parameters(self, derivatives, lr):
        db_f, dW_f, db_i, dW_i, db_c, dW_c, db_o, dW_o, db_y, dW_y = derivatives
        self.b_f -= lr * db_f
        self.W_f -= lr * dW_f
        self.b_i -= lr * db_i
        self.W_i -= lr * dW_i
        self.b_c -= lr * db_c
        self.W_c -= lr * dW_c
        self.b_o -= lr * db_o
        self.W_o -= lr * dW_o
        self.b_y -= lr * db_y
        self.W_y -= lr * dW_y
