import numpy as np
import os
import struct
import h5py
# import math


def load_signs_data():
    train_dataset = h5py.File('train_data/hand_sign/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('train_data/hand_sign/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    X_train = train_set_x_orig / 255.
    X_train = X_train.transpose(3, 1, 2, 0)
    X_test = test_set_x_orig / 255.
    X_test = X_test.transpose(3, 1, 2, 0)

    y_train = convert_to_one_hot(train_set_y_orig, 6).T
    y_train = y_train.transpose(1, 0)
    y_test = convert_to_one_hot(test_set_y_orig, 6).T
    y_test = y_test.transpose(1, 0)

    return X_train, y_train, X_test, y_test


# public method
def load_mnist():
    X_train, y_train = _load_mnist(kind='train')
    X_train = X_train.transpose()
    y_train = convert_to_one_hot(y_train, 10)
    X_test, y_test = _load_mnist(kind='t10k')
    X_test = X_test.transpose()
    y_test = convert_to_one_hot(y_test, 10)
    return X_train, y_train, X_test, y_test


def compute_samemode_pad(in_width, in_height, filter_size, strides):
    # 先确定输出维度，记住是上取整
    stride_width, stride_height = strides
    filter_width, filter_height = filter_size
    # out_height = math.ceil(float(in_height) / float(stride_height))
    # out_width  = math.ceil(float(in_width) / float(stride_width))

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
    # Note that this is different from existing libraries such as cuDNN and Caffe,
    # which explicitly specify the number of padded pixels and always pad the same
    # number of pixels on both sides.
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return (pad_top, pad_bottom, pad_left, pad_right)


def _load_mnist(kind='train'):
    path = "train_data/MNIST"
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def calculate_accuracy(model, X, Y):
    m, n = X.shape
    Y_pre = model.predict(X)
    Y_pre = Y_pre == np.max(Y_pre, axis=0, keepdims=True)
    C = np.abs(Y_pre - Y)
    error = np.sum(C) / 2
    print(1 - error / n)


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
