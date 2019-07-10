import numpy as np
from .Utils import create_mask_from_window, distribute_value
from ..common.GeneralLayer import GeneralLayer


class PoolingLayer:
    # mode value:'max', 'average'
    def __init__(self, pre_layer, filter_size, pad=(0, 0, 0, 0), mode='max',
                 stride=(1, 1)):
        GeneralLayer.__init__(self, pre_layer)
        self.pad = pad
        self.mode = mode
        self.filter_size = filter_size
        self.stride = stride

    def forward_propagation(self, Z):
        x_left_pad, x_right_pad, y_top_pad, y_botm_pad = self.pad
        Z_pad = np.pad(Z, ((x_left_pad, x_right_pad), (y_top_pad, y_botm_pad),
                       (0, 0), (0, 0)), 'constant', constant_values=0)
        self.Z_o = Z_pad
        filter_width, filter_height = self.filter_size
        x_stride, y_stride = self.stride
        width, height, channel, m = self.Z_o.shape

        # Define the dimensions of the output
        a_width = int(1 + (width - filter_width) / x_stride)
        a_height = int(1 + (height - filter_height) / y_stride)
        A = np.zeros((a_width, a_height, channel, m))
        for i in range(m):
            for c in range(channel):
                for x in range(a_width):
                    for y in range(a_height):
                        # Find the corners of the current "slice"
                        x_start = x * x_stride
                        y_start = y * y_stride
                        x_end = x_start + filter_width
                        y_end = y_start + filter_height
                        z_slice = self.Z_o[x_start:x_end, y_start:y_end, c, i]
                        if self.mode == "max":
                            A[x, y, c, i] = np.max(z_slice)
                        elif self.mode == "average":
                            A[x, y, c, i] = np.mean(z_slice)
                        else:
                            pass
        return A

    def back_propagation(self, dA, *kargs, **kwargs):
        Z_o = self.Z_o
        a_width, a_height, channel, m = dA.shape
        dZ = np.zeros(Z_o.shape)
        filter_width, filter_height = self.filter_size
        x_stride, y_stride = self.stride

        for i in range(m):
            for x in range(a_width):
                for y in range(a_height):
                    for c in range(channel):
                        # Find the corners of the current "slice"
                        x_start = x * x_stride
                        y_start = y * y_stride
                        x_end = x_start + filter_width
                        y_end = y_start + filter_height
                        if self.mode == "max":
                            slice = Z_o[x_start:x_end, y_start:y_end, c, i]
                            mask = create_mask_from_window(slice)
                            dZ[x_start:x_end, y_start:y_end, c, i] += mask * \
                                dA[x, y, c, i]
                        elif self.mode == "average":
                            shape = (filter_width, filter_height)
                            dZ[x_start:x_end, y_start:y_end, c, i] += \
                                distribute_value(dA[x, y, c, i], shape)
                        else:
                            pass
        x_left_pad, x_right_pad, y_top_pad, y_botm_pad = self.pad
        if x_right_pad == 0 and y_botm_pad == 0:
            dZ = dZ[x_left_pad:, y_top_pad:, :, :]
        elif x_right_pad == 0 and y_botm_pad > 0:
            dZ = dZ[x_left_pad:, y_top_pad:-y_botm_pad, :, :]
        elif x_right_pad > 0 and y_botm_pad == 0:
            dZ = dZ[x_left_pad:-x_right_pad, y_top_pad:, :, :]
        else:
            dZ = dZ[x_left_pad:-x_right_pad, y_top_pad:-y_botm_pad, :, :]
        return dZ

    def update_parameters(self, learning_rate):
        pass
