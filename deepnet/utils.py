import numpy as np
import _pickle as cPickle
import gzip
import os
import h5py


def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_mnist(path, num_training=50000, num_test=10000, cnn=True, one_hot=False):
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = cPickle.load(
        f, encoding='iso-8859-1')
    f.close()
    X_train, y_train = training_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data
    if cnn:
        shape = (-1, 1, 28, 28)
        X_train = X_train.reshape(shape)
        X_validation = X_validation.reshape(shape)
        X_test = X_test.reshape(shape)
    if one_hot:
        y_train = one_hot_encode(y_train, 10)
        y_validation = one_hot_encode(y_validation, 10)
        y_test = one_hot_encode(y_test, 10)
    X_train, y_train = X_train[range(
        num_training)], y_train[range(num_training)]
    X_test, y_test = X_test[range(num_test)], y_test[range(num_test)]
    return (X_train, y_train), (X_test, y_test)


def load_cifar10(path, num_training=1000, num_test=1000):
    Xs, ys = [], []
    for batch in range(1, 6):
        f = open(os.path.join(path, "data_batch_{0}".format(batch)), 'rb')
        data = cPickle.load(f, encoding='iso-8859-1')
        f.close()
        X = data["data"].reshape(10000, 3, 32, 32).astype("float64")
        y = np.array(data["labels"])
        Xs.append(X)
        ys.append(y)
    f = open(os.path.join(path, "test_batch"), 'rb')
    data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    X_train, y_train = np.concatenate(Xs), np.concatenate(ys)
    X_test = data["data"].reshape(10000, 3, 32, 32).astype("float")
    y_test = np.array(data["labels"])
    X_train, y_train = X_train[range(
        num_training)], y_train[range(num_training)]
    X_test, y_test = X_test[range(num_test)], y_test[range(num_test)]
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train)
    X_train /= 255.0
    X_test /= 255.0
    return (X_train, y_train), (X_test, y_test)

def load_handsign():
    train_dataset = h5py.File('data/hand-signs/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/hand-signs/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    X_train = train_set_x_orig/255.
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = test_set_x_orig/255.
    X_test = X_test.transpose(0, 3, 1, 2)
    # y_train = one_hot_encode(train_set_y_orig.T, len(classes))
    # y_test = one_hot_encode(test_set_y_orig.T, len(classes))
    y_train = np.squeeze(train_set_y_orig)
    y_test = np.squeeze(test_set_y_orig)
    return (X_train, y_train), (X_test, y_test)
