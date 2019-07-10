import numpy as np


def create_mask_from_window(slice):
    mask = (slice == np.max(slice))
    return mask


def distribute_value(dz, shape):
    h, w = shape
    a = (np.ones(shape) * dz) / (h * w)
    return a


def compute_same_mode_pad(in_size, filter_size, strides):
    # 先确定输出维度，记住是上取整
    in_width, in_height = in_size
    stride_width, stride_height = strides
    filter_width, filter_height = filter_size

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

    return (pad_left, pad_right, pad_top, pad_bottom)
