import numpy as np
import math


def shuffle(X, y):
    n = y.shape[0]
    order = np.arange(n)
    np.random.shuffle(order)
    X = X[order]
    y = y[order]
    return (X, y)


def generate_batchs(X, y, batch_size):
    batches = []
    m = X.shape[-1]
    batch_num = math.floor(m / batch_size)

    # size of samples is the last axis, need revert to the first axis
    axis = np.arange(len(X.shape))
    last = axis[-1]
    new_axis = list(axis[:-1])
    new_axis.insert(0, last)
    X = X.transpose(*new_axis)
    y = y.transpose()
    X, y = shuffle(X, y)
    for k in range(0, batch_num):
        batch_X = X[k * batch_size:(k + 1) * batch_size]
        batch_Y = y[k * batch_size:(k + 1) * batch_size]
        # recover axis order
        axis = np.arange(len(batch_X.shape))
        first = axis[0]
        new_axis = list(axis[1:])
        new_axis.append(first)
        batch_X = batch_X.transpose(*new_axis)
        batch_Y = batch_Y.transpose()
        batches.append((batch_X, batch_Y))
    if m % batch_size != 0:
        batch_X = X[batch_num * batch_size:m]
        batch_Y = y[batch_num * batch_size:m]
        # recover axis order
        axis = np.arange(len(batch_X.shape))
        first = axis[0]
        new_axis = list(axis[1:])
        new_axis.append(first)
        batch_X = batch_X.transpose(*new_axis)
        batch_Y = batch_Y.transpose()
        batches.append((batch_X, batch_Y))
    return batches


def create_mask_from_window(slice):
    mask = (slice == np.max(slice))
    return mask


def distribute_value(dz, shape):
    h, w = shape
    a = (np.ones(shape) * dz) / (h * w)
    return a


def compute_same_mode_pad(in_size, filter_size, strides):
    # 先确定输出维度，记住是上取整
    in_height, in_width = in_size
    stride_height, stride_width = strides
    filter_height, filter_width = filter_size

    if (in_height % stride_height == 0):
        pad_along_height = max(filter_height - stride_height, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_height), 0)

    if (in_width % stride_width == 0):
        pad_along_width = max(filter_width - stride_width, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_width), 0)

    # 因为pad是在上下、左右四侧pad。所以当pi不为偶数时要分配下
    # 这里是当pi为奇数时，下侧比上侧多一，右侧比左侧多一。
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (pad_top, pad_bottom, pad_left, pad_right)
