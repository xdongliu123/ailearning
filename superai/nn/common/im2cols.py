import numpy as np


def get_im2col_indices(c, out_h, out_w, stride_h, stride_w, f_h, f_w):
    k = np.repeat(np.arange(c), f_h * f_w).reshape(-1, 1)
    i0 = np.tile(np.repeat(np.arange(f_h), f_w), c)
    i1 = np.repeat(np.arange(out_h) * stride_h, out_w)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j0 = np.tile(np.arange(f_w), f_h * c)
    j1 = np.tile(np.arange(out_w) * stride_w, out_h)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    return (i, j, k)


def im2cols(X, out_h, out_w, stride_h, stride_w, f_h, f_w):
    c = X.shape[0]
    i, j, k = get_im2col_indices(c, out_h, out_w, stride_h, stride_w, f_h, f_w)
    # xcol is shape of (c*f_h*f_w, out_h*out_w, n)
    xcol = X[k, i, j, :]
    return xcol.reshape(c * f_h * f_w, -1)


def col2im(cols, X_shape, out_h, out_w, stride_h, stride_w, f_h, f_w):
    c, h, w, n = X_shape
    X = np.zeros((c, h, w, n), dtype=cols.dtype)
    i, j, k = get_im2col_indices(c, out_h, out_w, stride_h, stride_w, f_h, f_w)
    cols_reshaped = cols.reshape(c * f_h * f_w, -1, n)
    np.add.at(X, (k, i, j, slice(None)), cols_reshaped)
    return X
