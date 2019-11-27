import numpy as np
from .generic import GeneralLayer
# from ..common.utils import create_mask_from_window, distribute_value
from ..common.im2cols import im2cols, col2im


class PoolingLayer(GeneralLayer):
    # mode: 'max' or 'average'
    # pad: determin the padding value for input.(h_top_pad, h_bottom_pad, w_left_pad, w_right_pad)
    # filter_size: (filter_height, filter_width)
    def __init__(self, filter_size, pad=(0, 0, 0, 0), mode='max', stride=(1, 1)):
        GeneralLayer.__init__(self)
        self.pad = pad
        self.mode = mode
        self.filter_size = filter_size
        self.stride = stride

    def forward_propagation(self, X):
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        self.X_pad = np.pad(X, ((0, 0), (h_top_pad, h_bottom_pad), (w_left_pad, w_right_pad),
                            (0, 0)), 'constant', constant_values=0)
        f_h, f_w = self.filter_size
        h_s, w_s = self.stride
        c, h, w, m = self.X_pad.shape

        # Define the dimensions of the output
        out_w = int(1 + (w - f_w) / w_s)
        out_h = int(1 + (h - f_h) / h_s)
        im = self.X_pad.transpose(3, 0, 1, 2)
        im = im.reshape(im.shape[0] * im.shape[1], 1, im.shape[2], im.shape[3])
        im = im.transpose(1, 2, 3, 0)
        self.cols = im2cols(im, out_h, out_w, h_s, w_s, f_h, f_w)
        if self.mode == "max":
            self.max_indexes = np.argmax(self.cols, axis=0)
            out = self.cols[self.max_indexes, range(self.max_indexes.size)]
        elif self.mode == "average":
            out = np.mean(self.cols, axis=0)
        else:
            self.max_indexes = np.argmax(self.cols, axis=0)
            out = self.cols[self.max_indexes, range(self.max_indexes.size)]
        out = out.reshape(out_h, out_w, m, c).transpose(3, 0, 1, 2)
        return out

    def back_propagation(self, dA, l1_lambd, l2_lambd):
        c, out_h, out_w, m = dA.shape
        f_h, f_w = self.filter_size
        h_s, w_s = self.stride
        dx_col = np.zeros(self.cols.shape)
        dx_col[self.max_indexes, range(self.max_indexes.size)] = dA.transpose(1, 2, 3, 0).ravel()
        shape = (1, self.X_pad.shape[1], self.X_pad.shape[2], self.X_pad.shape[0] * m)
        dx = col2im(dx_col, shape, out_h, out_w, h_s, w_s, f_h, f_w)
        dx = dx.reshape(self.X_pad.shape[1], self.X_pad.shape[2], m, self.X_pad.shape[0])
        dx = dx.transpose(3, 0, 1, 2)
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        return dx[:, h_top_pad:None if h_bottom_pad == 0 else -h_bottom_pad,
                  w_left_pad:None if w_right_pad == 0 else -w_right_pad, :]

    '''
    # teaching version
    # very slow
    def forward_propagation(self, X):
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        self.X_pad = np.pad(X, ((0, 0), (h_top_pad, h_bottom_pad), (w_left_pad, w_right_pad),
                            (0, 0)), 'constant', constant_values=0)
        f_h, f_w = self.filter_size
        h_s, w_s = self.stride
        c, h, w, m = self.X_pad.shape

        # Define the dimensions of the output
        out_w = int(1 + (w - f_w) / w_s)
        out_h = int(1 + (h - f_h) / h_s)
        A = np.zeros((c, out_h, out_w, m))
        for i in range(m):
            for j in range(c):
                for x in range(out_w):
                    for y in range(out_h):
                        # Find the corners of the current "slice"
                        x_start = x * w_s
                        y_start = y * h_s
                        x_end = x_start + f_w
                        y_end = y_start + f_h
                        z_slice = self.X_pad[j, y_start:y_end, x_start:x_end, i]
                        if self.mode == "max":
                            A[j, y, x, i] = np.max(z_slice)
                        elif self.mode == "average":
                            A[j, y, x, i] = np.mean(z_slice)
                        else:
                            pass
        return A

    def back_propagation(self, dA, l1_lambd, l2_lambd):
        c, h, w, m = dA.shape
        dZ = np.zeros(self.X_pad.shape)
        f_h, f_w = self.filter_size
        h_s, w_s = self.stride

        for i in range(m):
            for x in range(w):
                for y in range(h):
                    for j in range(c):
                        # Find the corners of the current "slice"
                        x_start = x * w_s
                        y_start = y * h_s
                        x_end = x_start + f_w
                        y_end = y_start + f_h
                        if self.mode == "max":
                            slice = self.X_pad[j, y_start:y_end, x_start:x_end, i]
                            mask = create_mask_from_window(slice)
                            dZ[j, y_start:y_end, x_start:x_end, i] += mask * dA[j, y, x, i]
                        elif self.mode == "average":
                            shape = (f_h, f_w)
                            dZ[j, y_start:y_end, x_start:x_end, i] += \
                                distribute_value(dA[j, y, x, i], shape)
                        else:
                            pass
        h_top_pad, h_bottom_pad, w_left_pad, w_right_pad = self.pad
        return dZ[:, h_top_pad:None if h_bottom_pad == 0 else -h_bottom_pad,
                  w_left_pad:None if w_right_pad == 0 else -w_right_pad, :]
    '''
