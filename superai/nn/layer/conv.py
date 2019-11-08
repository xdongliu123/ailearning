import numpy as np
from .generic import GeneralLayer
from ..common.im2cols import im2cols, col2im


class Conv(GeneralLayer):
    # pad: determin the padding value for input.(h_top_pad, h_bottom_pad, w_left_pad, w_right_pad)
    # filter_size: (filter_channel, filter_height, filter_width)
    def __init__(self, filter_size, filter_num, pad=(0, 0, 0, 0), stride=(1, 1)):
        GeneralLayer.__init__(self)
        self.pad = pad
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.filter_stride = stride
        # init parameters
        # np.random.seed(1)
        param_size = (filter_num, ) + filter_size
        self.W = np.random.randn(*param_size) / np.sqrt(filter_num / 2.)
        self.b = np.zeros((filter_num, 1))  # used for im2col ver of forward and backward methods
        # self.b = np.zeros((filter_num, 1, 1, 1))  # used for demo ver of forward and backward
        self.params.append(self.W)
        self.params.append(self.b)

    # 'X' is input of this layer
    def forward_propagation(self, X):
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        self.X_pad = np.pad(X, ((0, 0), (h_top_pad, h_bottom_pad), (w_left_pad, w_right_pad),
                            (0, 0)), 'constant', constant_values=0)
        c, h, w, m = self.X_pad.shape
        f_c, f_h, f_w = self.filter_size
        assert f_w <= w and f_h <= h, "filter size couldn't be larger than X's width and height"
        assert f_c == c, "input channel's doesn't match filter's channel"

        h_s, w_s = self.filter_stride
        out_w = 1 + int((w - f_w) / w_s)
        out_h = 1 + int((h - f_h) / h_s)

        cols = im2cols(self.X_pad, out_h, out_w, h_s, w_s, f_h, f_w)
        self.cols = cols
        weights = self.W.reshape(self.filter_num, -1)
        Z = weights @ cols + self.b
        self.Z = Z.reshape(self.filter_num, out_h, out_w, m)
        return self.Z

    def back_propagation(self, dZ):
        out_c, out_h, out_w,  m = dZ.shape
        h_s, w_s = self.filter_stride
        f_c, f_h, f_w = self.filter_size
        # convert dZ to shape of (out_c, out_h * out_w * m)
        dZ_cols = dZ.reshape(out_c, -1)
        weights = self.W.reshape(self.filter_num, -1)
        # dcols is shape of (f_c * f_h * f_w, out_h * out_w * m)
        dcols = weights.T @ dZ_cols
        dX = col2im(dcols, self.X_pad.shape, out_h, out_w, h_s, w_s, f_h, f_w)

        dW = dZ_cols @ self.cols.T
        self.dW = dW.reshape(self.W.shape)
        self.db = np.sum(dZ, axis=(1, 2, 3)).reshape(self.filter_num, -1)
        self.grads = [self.dW, self.db]

        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        dX = dX[:, h_top_pad:None if h_bottom_pad == 0 else -h_bottom_pad,
                w_left_pad:None if w_right_pad == 0 else -w_right_pad, :]
        return dX
    '''
    # teach version for easy understanding
    # very slow
    def forward_propagation(self, X):
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        self.X_pad = np.pad(X, ((0, 0), (h_top_pad, h_bottom_pad), (w_left_pad, w_right_pad),
                            (0, 0)), 'constant', constant_values=0)
        c, h, w, m = self.X_pad.shape
        f_c, f_h, f_w = self.filter_size
        assert f_w <= w and f_h <= h, "filter size couldn't be larger than X's width and height"
        assert f_c == c, "input channel's doesn't match filter's channel"

        h_s, w_s = self.filter_stride
        out_w = 1 + int((w - f_w) / w_s)
        out_h = 1 + int((h - f_h) / h_s)
        out_c = self.filter_num
        self.Z = np.zeros((out_c, out_h, out_w, m))

        for i in range(m):  # loop over the batch of training examples
            for j in range(out_c):
                W_c = self.W[j, :, :, :]
                b_c = self.b[j, :, :, :]
                for y in range(out_h):
                    for x in range(out_w):
                        # Find the corners of the current "slice"
                        x_start = x * w_s
                        y_start = y * h_s
                        x_end = x_start + f_w
                        y_end = y_start + f_h
                        # Use the corners to define the slice of X_pad
                        slice = self.X_pad[:, y_start:y_end, x_start:x_end, i]
                        # Convolve the (3D) slice with the correct filter W and bias b
                        self.Z[j, y, x, i] = np.sum(np.multiply(slice, W_c) + b_c)
        return self.Z

    def back_propagation(self, dZ):
        z_c, z_h, z_w,  m = dZ.shape
        h_s, w_s = self.filter_stride
        f_c, f_h, f_w = self.filter_size

        dX_pad = np.zeros(self.X_pad.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        for i in range(m):
            for j in range(z_c):
                for x in range(z_w):
                    for y in range(z_h):
                        # Find the corners of the current "slice"
                        x_start = x * w_s
                        y_start = y * h_s
                        x_end = x_start + f_w
                        y_end = y_start + f_h

                        dz = dZ[j, y, x, i]
                        slice = self.X_pad[:, y_start:y_end, x_start:x_end, i]
                        dx = np.multiply(self.W[j, :, :, :], dz)
                        dX_pad[:, y_start:y_end, x_start:x_end, i] += dx
                        dW[j, :, :, :] += np.multiply(slice, dz)
                        db[j, :, :, :] += dz
        self.dW = dW
        self.db = db
        self.grads = [self.dW, self.db]

        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        d_X = dX_pad[:, h_top_pad:None if h_bottom_pad == 0 else -h_bottom_pad,
                     w_left_pad:None if w_right_pad == 0 else -w_right_pad, :]
        return d_X
    '''
