import numpy as np
from ..common.GeneralLayer import GeneralLayer
from ..common.Utils import tanh, tanh_derivative, relu, relu_derivative


class CNNLayer(GeneralLayer):
    # 'pad' determin the padding value of pre_A's width and height dimension
    # (x_pad, y_pad)
    def __init__(self, pre_layer, filter_size, filter_num, pad=(0, 0, 0, 0),
                 stride=(1, 1), activator=None):
        GeneralLayer.__init__(self, pre_layer)
        filter_width, filter_height, filter_channel = filter_size
        self.pad = pad
        self.W = np.random.randn(filter_width, filter_height, filter_channel,
                                 filter_num)
        self.b = np.random.randn(1, 1, 1, filter_num)
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.filter_stride = stride
        self.activator = activator

    # 'X' is input of this layer
    def forward_propagation(self, X, *kargs, **kwargs):
        x_left_pad, x_right_pad, y_top_pad, y_botm_pad = self.pad
        X_pad = np.pad(X, ((x_left_pad, x_right_pad), (y_top_pad, y_botm_pad),
                       (0, 0), (0, 0)), 'constant', constant_values=0)
        self.X_pad = X_pad

        width, height, channel, m = X_pad.shape
        filter_width, filter_height, filter_channel = self.filter_size
        assert filter_width <= width and filter_height <= height, "filter size \
        couldn't be larger than X's width and height"
        assert filter_channel == channel, "X's depth doesn't match current \
        layer's filter channel"

        x_stride, y_stride = self.filter_stride
        out_width = 1 + int((width - filter_width) / x_stride)
        out_height = 1 + int((height - filter_height) / y_stride)
        out_depth = self.filter_num
        Z = np.zeros((out_width, out_height, out_depth, m))
        for i in range(m):  # loop over the batch of training examples
            for c in range(out_depth):
                W_c = self.W[:, :, :, c]
                b_c = self.b[:, :, :, c]
                for x in range(out_width):
                    for y in range(out_height):
                        # Find the corners of the current "slice"
                        x_start = x * x_stride
                        y_start = y * y_stride
                        x_end = x_start + filter_width
                        y_end = y_start + filter_height
                        # Use the corners to define the slice of a_prev_pad
                        slice = X_pad[x_start:x_end, y_start:y_end, :, i]
                        # Convolve the (3D) slice with the correct filter W and
                        # bias b
                        Z[x, y, c, i] = np.sum(np.multiply(slice, W_c) + b_c)
        if self.activator == "tanh":
            Z = tanh(Z)
        elif self.activator == "relu":
            Z = relu(Z)
        else:
            pass

        self.Z = Z
        return Z

    def back_propagation(self, dZ, *kargs, **kwargs):
        z_width, z_height, filter_num, m = dZ.shape
        x_stride, y_stride = self.filter_stride
        filter_width, filter_height, filter_channel = self.filter_size

        if self.activator == "tanh":
            dZ = tanh_derivative(self.Z) * dZ
        elif self.activator == "relu":
            dZ = relu_derivative(self.Z) * dZ
        else:
            pass

        dX_pad = np.zeros(self.X_pad.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        for i in range(m):
            for c in range(filter_num):
                for x in range(z_width):
                    for y in range(z_height):
                        # Find the corners of the current "slice"
                        x_start = x * x_stride
                        y_start = y * y_stride
                        x_end = x_start + filter_width
                        y_end = y_start + filter_height

                        dz = dZ[x, y, c, i]
                        slice = self.X_pad[x_start:x_end, y_start:y_end, :, i]
                        dx = np.multiply(self.W[:, :, :, c], dz)
                        dX_pad[x_start:x_end, y_start:y_end, :, i] += dx
                        dW[:, :, :, c] += (np.multiply(slice, dz))
                        db[:, :, :, c] += dz
        self.dW = dW
        self.db = db
        x_left_pad, x_right_pad, y_top_pad, y_botm_pad = self.pad
        if x_right_pad == 0 and y_botm_pad == 0:
            d_X = dX_pad[x_left_pad:, y_top_pad:, :, :]
        elif x_right_pad == 0 and y_botm_pad > 0:
            d_X = dX_pad[x_left_pad:, y_top_pad:-y_botm_pad, :, :]
        elif x_right_pad > 0 and y_botm_pad == 0:
            d_X = dX_pad[x_left_pad:-x_right_pad, y_top_pad:, :, :]
        else:
            d_X = dX_pad[x_left_pad:-x_right_pad, y_top_pad:-y_botm_pad, :, :]

        return d_X

    def update_parameters(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
